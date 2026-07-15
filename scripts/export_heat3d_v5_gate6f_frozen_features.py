#!/usr/bin/env python3
"""Cache N3 e402 train/valid inference features for Gate 6F scale probes.

The script is deliberately inference-only: it materializes only ``train`` and
``valid_iid`` groups, runs no gradient calculation, and writes an ignored cache
outside every existing N3/V13 run directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
from typing import Any

import jax
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from check_heat3d_v4_registry import resolve_inherited_yaml  # noqa: E402
from rigno.heat3d_v5_global_context import GLOBAL_CONTEXT_FEATURES  # noqa: E402
from rigno.heat3d_v5_scale_pooling import QK_REGION_FEATURES  # noqa: E402
from rigno.heat3d_v5_shape_scale import target_shape_scale  # noqa: E402
from scripts import run_heat3d_v4_controlled_training as v4_wrapper  # noqa: E402


N3_ID = "V4P5_07_native_pooled_latent_global_film"
MODEL_METADATA_ONLY_FIELDS = {
    "architecture",
    "report_memory_estimate",
    "report_parameter_count",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/heat3d_v5/generated/V4P5_07_native_pooled_latent_global_film.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Read-only N3 params_best.pkl (epoch 402).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Ignored output/ directory dedicated to Gate 6F feature cache.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sample_ids_hash(sample_ids: list[str]) -> str:
    digest = hashlib.sha256()
    for sample_id in sample_ids:
        digest.update(sample_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _resolved_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: YAML root must be a mapping")
    config_id = payload.get("config_id")
    if not isinstance(config_id, str) or not config_id:
        raise ValueError(f"{path}: inherited Gate 6F source YAML needs a config_id")
    resolved = resolve_inherited_yaml(payload, path)
    # The generic V4 inheritance resolver intentionally produces the runnable
    # payload only and drops source-only identity fields.  Gate 6F binds a
    # frozen cache to its source config, so retain that declared identity as
    # provenance after resolution rather than inferring it from a path.
    resolved["config_id"] = config_id
    return resolved


def _root_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _ensure_cache_dir(path: Path) -> Path:
    resolved = path.resolve()
    output_root = (ROOT / "output").resolve()
    if output_root not in resolved.parents and resolved != output_root:
        raise ValueError(f"cache dir must remain under output/: {path}")
    if "V4P5_07_native_pooled_latent_global_film" in str(resolved):
        raise ValueError("Gate 6F cache must not write inside the N3 run directory")
    if "V4P5_13_gate6e_scratch_branch_rebalance" in str(resolved):
        raise ValueError("Gate 6F cache must not write inside the V13 run directory")
    resolved.mkdir(parents=True, exist_ok=False)
    return resolved


def _install_n3_semantics(config: dict[str, Any]) -> Any:
    dataset = config["dataset"]
    v4_wrapper._install_profile_hooks(
        str(dataset["normalization_profile"]),
        str(dataset["condition_feature_transform"]),
        str(dataset["input_feature_schema"]),
        str(dataset["coord_policy"]),
        str(dataset["extent_feature_policy"]),
    )
    return v4_wrapper.legacy_runner


def _runtime_model_source(runner: Any, declared: dict[str, Any]) -> dict[str, Any]:
    """Mirror the runner's CLI whitelist before constructing RIGNO."""

    accepted = set(runner.GraphNeuralOperator.__dataclass_fields__)
    unsupported = set(declared) - accepted - MODEL_METADATA_ONLY_FIELDS
    if unsupported:
        raise ValueError(
            "Gate 6F cache model fields are neither RIGNO inputs nor known metadata: "
            f"{sorted(unsupported)}"
        )
    model_source = dict(runner.RUNNER_MODEL_CONFIG)
    model_source.update({key: value for key, value in declared.items() if key in accepted})
    # N3 predates the Gate 6F switches, so these fields are intentionally
    # absent from its YAML and supplied by argparse in the normal runner path.
    # The cache exporter bypasses argparse; mirror the exact disabled defaults
    # before validating or constructing the frozen model.
    model_source.setdefault("scale_head_depth", 1)
    model_source.setdefault("pooled_latent_stop_gradient", False)
    return model_source


def _build_train_valid_groups(config: dict[str, Any]):
    """Build only train/valid groups; forbidden roles are never materialized."""

    runner = _install_n3_semantics(config)
    dataset_config = config["dataset"]
    run_config = config["run"]
    sample_root = _root_path(str(dataset_config["subset_path"]))
    split_map = _root_path(str(dataset_config["split_map_path"]))
    split_ids, split_source, primary_split, _ = runner._resolve_training_splits(
        sample_root, split_map
    )
    if primary_split != "valid_iid":
        raise ValueError(f"Gate 6F requires valid_iid, got {primary_split!r}")
    train_ids = list(split_ids["train"])
    valid_ids = list(split_ids["valid_iid"])
    if len(train_ids) != 672 or len(valid_ids) != 128:
        raise ValueError(
            f"Gate 6F requires train=672/valid_iid=128, got {len(train_ids)}/{len(valid_ids)}"
        )
    dataset = runner.Heat3DV1NativeSupervisedDataset(
        sample_root,
        k_encoding_mode="diag3",
        boundary_mask_fallback=False,
    )
    index_by_id = dataset.sample_index_by_id()
    train_examples = [dataset[index_by_id[sample_id]] for sample_id in train_ids]
    valid_examples = [dataset[index_by_id[sample_id]] for sample_id in valid_ids]
    stats = runner._train_only_stats(train_examples)
    model_source = _runtime_model_source(runner, config["model"])
    model_config = runner._resolve_decoder_bypass_model_config(model_source, stats)
    runner._validate_model_config(model_config)
    graph_config = config["graph"]
    builder = runner.Heat3DGraphBuilder(**graph_config)
    train_groups = runner._make_groups_with_progress(
        train_examples,
        stats,
        builder,
        "train",
        False,
        "basic",
        int(config["optimizer"]["graph_seed"]),
        batch_size=int(run_config["batch_size"]),
        drop_last=False,
    )
    valid_groups = runner._make_groups_with_progress(
        valid_examples,
        stats,
        builder,
        "valid_iid",
        False,
        "basic",
        int(config["optimizer"]["graph_seed"]),
        batch_size=int(run_config["validation_batch_size"]),
        drop_last=False,
    )
    encoded_context, context_payload = runner._prepare_global_context_lookup(
        model_config,
        train_examples=train_examples,
        required_examples=valid_examples,
    )
    for groups in (train_groups, valid_groups):
        runner._attach_global_context_to_groups(
            groups,
            encoded_context,
            expected_feature_dim=int(model_config["global_context_feature_dim"]),
        )
    examples_by_id = {example.sample_id: example for example in [*train_examples, *valid_examples]}
    for groups in (train_groups, valid_groups):
        runner._attach_native_physics_to_groups(groups, examples_by_id)
        # Cache the raw q--k features for later static probes even though the
        # frozen N3 mean-pool model itself does not consume them.
        runner._attach_qk_region_features_to_groups(groups, examples_by_id)
    raw_context_by_id = {}
    for example in [*train_examples, *valid_examples]:
        row = runner._global_context_row_for_example(example)
        raw_context_by_id[str(example.sample_id)] = np.asarray(
            [row[name] for name in GLOBAL_CONTEXT_FEATURES], dtype=np.float32
        )
    return {
        "runner": runner,
        "model_config": model_config,
        "stats": stats,
        "train_groups": train_groups,
        "valid_groups": valid_groups,
        "train_ids": train_ids,
        "valid_ids": valid_ids,
        "split_source": split_source,
        "context_payload": context_payload,
        "raw_context_by_id": raw_context_by_id,
    }


def _to_host(value: Any) -> np.ndarray:
    return np.asarray(jax.device_get(value))


def _cache_split(
    *,
    runner: Any,
    model: Any,
    params: Any,
    groups: list[dict],
    raw_context_by_id: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], list[str]]:
    payloads: dict[str, list[np.ndarray]] = {
        "global_context": [],
        "global_context_raw": [],
        "rnodes_processed_pre_film": [],
        "rnodes_processed": [],
        "qk_region_features": [],
        "phi_hat": [],
        "s_phys": [],
        "s_true": [],
        "s_hat_n3": [],
        "target_deltaT": [],
        "control_volumes": [],
        "dirichlet_mask": [],
        "dirichlet_delta": [],
    }
    sample_ids: list[str] = []
    for group in groups:
        prediction = runner._model_apply(model, params, group)
        physics = group["native_physics"]
        s_true, _ = target_shape_scale(
            group["target_delta_raw"],
            physics["control_volumes"],
            dirichlet_mask=physics["dirichlet_mask"],
        )
        sample_ids.extend(str(value) for value in group["sample_ids"])
        payloads["global_context"].append(_to_host(group["global_context"]))
        payloads["global_context_raw"].append(
            np.stack([raw_context_by_id[str(sample_id)] for sample_id in group["sample_ids"]])
        )
        payloads["rnodes_processed_pre_film"].append(
            _to_host(prediction["rnodes_processed_pre_film"])
        )
        payloads["rnodes_processed"].append(_to_host(prediction["rnodes_processed"]))
        payloads["qk_region_features"].append(_to_host(group["qk_region_features"]))
        payloads["phi_hat"].append(_to_host(prediction["phi_hat"])[:, 0, :, 0])
        payloads["s_phys"].append(np.exp(_to_host(physics["log_s_phys"])))
        payloads["s_true"].append(_to_host(s_true)[:, 0, 0, 0])
        payloads["s_hat_n3"].append(_to_host(prediction["s_hat"])[:, 0, 0, 0])
        payloads["target_deltaT"].append(_to_host(group["target_delta_raw"])[:, 0, :, 0])
        payloads["control_volumes"].append(_to_host(physics["control_volumes"]))
        payloads["dirichlet_mask"].append(_to_host(physics["dirichlet_mask"]))
        prescribed = _to_host(physics["prescribed_temperature"])
        reference = _to_host(physics["reference_temperature"])
        payloads["dirichlet_delta"].append(prescribed - reference)
    arrays = {key: np.concatenate(values, axis=0) for key, values in payloads.items()}
    if arrays["phi_hat"].shape[0] != len(sample_ids):
        raise AssertionError("frozen cache sample count mismatch")
    if not all(np.all(np.isfinite(value)) for value in arrays.values()):
        raise ValueError("frozen cache contains non-finite values")
    return arrays, sample_ids


def _default_path_replay_audit(
    *,
    runner: Any,
    model_config: dict[str, Any],
    checkpoint_path: Path,
    train_group: dict,
    seed: int,
) -> dict[str, Any]:
    """Prove explicit disabled controls exactly replay the N3 default path."""

    base_model = runner.GraphNeuralOperator(**model_config)
    explicit_config = dict(model_config)
    explicit_config.update(
        {
            "scale_pooling": "mean",
            "scale_head_depth": 1,
            "pooled_latent_stop_gradient": False,
        }
    )
    explicit_model = runner.GraphNeuralOperator(**explicit_config)
    key = jax.random.PRNGKey(seed)
    base_initial = runner._model_init(base_model, key, train_group, train_group["inputs"])["params"]
    explicit_initial = runner._model_init(
        explicit_model, key, train_group, train_group["inputs"]
    )["params"]
    base_params, _ = runner._load_init_checkpoint_params(
        base_initial, checkpoint_path, strict=True, partial_load_policy="matching"
    )
    explicit_params, _ = runner._load_init_checkpoint_params(
        explicit_initial, checkpoint_path, strict=True, partial_load_policy="matching"
    )
    base_items = {path: np.asarray(value) for path, value in runner._param_leaf_items(base_params)}
    explicit_items = {
        path: np.asarray(value) for path, value in runner._param_leaf_items(explicit_params)
    }
    same_keys = set(base_items) == set(explicit_items)
    parameter_max_abs = max(
        [
            float(np.max(np.abs(base_items[path] - explicit_items[path])))
            for path in sorted(base_items)
        ]
        or [0.0]
    ) if same_keys else float("inf")
    base_prediction = runner._model_apply(base_model, base_params, train_group)
    explicit_prediction = runner._model_apply(explicit_model, explicit_params, train_group)
    output_fields = ("deltaT_hat", "phi_hat", "s_hat", "pooled_rnodes")
    output_max_abs = {
        field: float(
            np.max(
                np.abs(
                    _to_host(base_prediction[field]) - _to_host(explicit_prediction[field])
                )
            )
        )
        for field in output_fields
    }
    passed = bool(same_keys and parameter_max_abs == 0.0 and max(output_max_abs.values()) == 0.0)
    if not passed:
        raise AssertionError("disabled Gate 6F controls failed exact N3 replay")
    return {
        "passed": True,
        "explicit_disabled_fields": {
            "scale_pooling": "mean",
            "scale_head_depth": 1,
            "pooled_latent_stop_gradient": False,
            "scale_head_lr_multiplier": 1.0,
        },
        "parameter_leaf_keys_identical": same_keys,
        "parameter_max_abs_difference": parameter_max_abs,
        "output_max_abs_difference": output_max_abs,
    }


def main() -> int:
    args = _parse_args()
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    config = _resolved_config(config_path)
    if config.get("config_id") != N3_ID:
        raise ValueError(f"Gate 6F cache must use {N3_ID}, got {config.get('config_id')!r}")
    if config["model"].get("native_output_mode") != "native_shape_scale":
        raise ValueError("Gate 6F frozen cache requires native shape--scale N3")
    cache_dir = _ensure_cache_dir(args.cache_dir)
    assembled = _build_train_valid_groups(config)
    runner = assembled["runner"]
    checkpoint_payload = runner._load_params_checkpoint(checkpoint_path)
    if checkpoint_payload.get("checkpoint_kind") != "best" or int(checkpoint_payload.get("epoch", -1)) != 402:
        raise ValueError("Gate 6F cache requires N3 params_best.pkl at epoch 402")
    model = runner.GraphNeuralOperator(**assembled["model_config"])
    init_params = runner._model_init(
        model,
        jax.random.PRNGKey(int(config["optimizer"]["model_seed"])),
        assembled["train_groups"][0],
        assembled["train_groups"][0]["inputs"],
    )["params"]
    params, load_info = runner._load_init_checkpoint_params(
        init_params,
        checkpoint_path,
        strict=True,
        partial_load_policy="matching",
    )
    if not load_info["loaded"]:
        raise AssertionError("N3 cache checkpoint did not load")
    default_path_replay = _default_path_replay_audit(
        runner=runner,
        model_config=assembled["model_config"],
        checkpoint_path=checkpoint_path,
        train_group=assembled["train_groups"][0],
        seed=int(config["optimizer"]["model_seed"]),
    )
    split_records = {}
    for split, groups in (
        ("train", assembled["train_groups"]),
        ("valid_iid", assembled["valid_groups"]),
    ):
        arrays, sample_ids = _cache_split(
            runner=runner,
            model=model,
            params=params,
            groups=groups,
            raw_context_by_id=assembled["raw_context_by_id"],
        )
        cache_path = cache_dir / f"{split}_n3_e402_frozen_features.npz"
        np.savez_compressed(cache_path, sample_ids=np.asarray(sample_ids), **arrays)
        split_records[split] = {
            "sample_count": int(len(sample_ids)),
            "sample_ids_sha256": _sample_ids_hash(sample_ids),
            "artifact": str(cache_path),
            "artifact_sha256": _sha256(cache_path),
            "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
        }
    manifest = {
        "schema_version": "heat3d_v5_gate6f_frozen_feature_cache_v1",
        "config_id": N3_ID,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_kind": checkpoint_payload.get("checkpoint_kind"),
        "checkpoint_epoch": int(checkpoint_payload.get("epoch")),
        "checkpoint_git_commit": checkpoint_payload.get("git_commit"),
        "checkpoint_strict_load": load_info,
        "default_disabled_control_replay": default_path_replay,
        "roles_materialized": ["train", "valid_iid"],
        "forbidden_roles_materialized": [],
        "sealed_iid_accessed": False,
        "training_started": False,
        "gnn_backward": False,
        "global_context_feature_names": list(GLOBAL_CONTEXT_FEATURES),
        "qk_region_feature_names": list(QK_REGION_FEATURES),
        "global_context_standardizer": assembled["context_payload"]["standardizer"],
        "global_context_fit_roles": ["train"],
        "split_source": assembled["split_source"],
        "splits": split_records,
        "peak_rss_mb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0,
    }
    manifest_path = cache_dir / "cache_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "manifest": str(manifest_path), **manifest}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
