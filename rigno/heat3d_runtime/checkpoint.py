"""Checkpoint interpretation for the stable V7 inference runtime.

This module only loads the already-produced parameter artifact and materializes
its train-only normalization payload.  It does not initialize a model, read a
dataset, or mutate any other module's state.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from rigno.heat3d_v1_training_semantics import (
    decoder_bypass_required_full_condition_features,
)


DECODER_BYPASS_MODE_NONE = "none"
DECODER_BYPASS_FEATURES_FULL_CONDITION = "full_condition"
DECODER_BYPASS_FEATURES_EXPLICIT_LOCAL_CONDITION = "explicit_local_condition"


def file_sha256(path: str | Path) -> str:
    """Return the SHA256 of a checkpoint or other immutable artifact."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def materialize_checkpoint_stats(
    checkpoint_stats: Mapping[str, Any],
) -> dict[str, Any]:
    """Materialize the frozen stats payload without changing its values."""

    stats = dict(checkpoint_stats)
    stats["feature_names"] = tuple(stats.get("feature_names") or ())
    if stats.get("condition_feature_transforms"):
        stats["condition_feature_transforms"] = tuple(
            stats["condition_feature_transforms"]
        )
    required = (
        "target_delta_mean",
        "target_delta_std",
        "condition_mean",
        "condition_std",
    )
    missing = [name for name in required if name not in stats]
    if missing:
        raise ValueError(f"checkpoint normalization is missing fields: {missing}")
    stats["target_delta_mean"] = jnp.asarray(
        np.asarray(stats["target_delta_mean"], dtype=np.float32).reshape(1, 1, 1, 1)
    )
    stats["target_delta_std"] = jnp.asarray(
        np.asarray(stats["target_delta_std"], dtype=np.float32).reshape(1, 1, 1, 1)
    )
    stats["condition_mean"] = jnp.asarray(
        np.asarray(stats["condition_mean"], dtype=np.float32).reshape(1, 1, 1, -1)
    )
    stats["condition_std"] = jnp.asarray(
        np.asarray(stats["condition_std"], dtype=np.float32).reshape(1, 1, 1, -1)
    )
    if stats["condition_mean"].shape[-1] != len(stats["feature_names"]):
        raise ValueError("checkpoint condition stats do not match feature_names")
    return stats


def resolve_model_config(
    checkpoint_model_config: Mapping[str, Any],
    stats: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve checkpoint-dependent decoder-bypass feature indices.

    The old runtime resolved these indices after installing a V3 hook.  The
    same resolution is now a pure function over the checkpoint config and
    train-only stats, so no runner module needs to be patched.
    """

    resolved = dict(checkpoint_model_config)
    mode = str(resolved.get("decoder_bypass_mode", DECODER_BYPASS_MODE_NONE))
    if mode == DECODER_BYPASS_MODE_NONE:
        resolved["decoder_bypass_feature_indices"] = ()
        resolved["decoder_bypass_feature_names"] = ()
        resolved["decoder_bypass_num_features"] = 0
        return resolved

    feature_names = tuple(stats.get("feature_names") or ())
    feature_mode = str(resolved.get("decoder_bypass_features", "none"))
    if feature_mode == DECODER_BYPASS_FEATURES_FULL_CONDITION:
        required = decoder_bypass_required_full_condition_features(
            input_feature_schema=str(stats.get("input_feature_schema", "legacy_bc_flags")),
            extent_feature_policy=str(stats.get("extent_feature_policy", "none")),
            dual_robin="bottom_h" in feature_names,
        )
        missing = [name for name in required if name not in feature_names]
        if missing:
            raise ValueError(
                "decoder bypass missing condition features: "
                f"{missing}; available={feature_names}"
            )
        selected = feature_names
    elif feature_mode == DECODER_BYPASS_FEATURES_EXPLICIT_LOCAL_CONDITION:
        selected = tuple(resolved.get("decoder_bypass_local_feature_names") or ())
        if not selected:
            raise ValueError(
                "explicit_local_condition requires decoder_bypass_local_feature_names"
            )
        allowlist = {
            "k_x",
            "k_y",
            "k_z",
            "q",
            "is_top",
            "is_bottom",
            "is_side",
            "is_interior",
            "top_h",
            "bottom_h",
            "top_T_inf_minus_T_ref",
            "bottom_T_fixed_minus_T_ref",
            "bottom_T_inf_minus_T_ref",
        }
        disallowed = [name for name in selected if name not in allowlist]
        if disallowed:
            raise ValueError(
                "explicit local bypass contains unaudited features: "
                f"{disallowed}"
            )
        if len(selected) != len(set(selected)):
            raise ValueError("explicit local bypass feature names must be unique")
    else:
        raise ValueError(f"unsupported decoder_bypass_features={feature_mode!r}")

    missing = [name for name in selected if name not in feature_names]
    if missing:
        raise ValueError(
            "decoder bypass feature names are absent from checkpoint stats: "
            f"{missing}; available={feature_names}"
        )
    indices = tuple(feature_names.index(name) for name in selected)
    resolved["decoder_bypass_feature_indices"] = indices
    resolved["decoder_bypass_feature_names"] = tuple(selected)
    resolved["decoder_bypass_num_features"] = len(indices)
    return resolved


def device_params(params: Any) -> Any:
    """Place a parameter tree on the active JAX device without mutation."""

    return jax.tree_util.tree_map(
        lambda value: jnp.asarray(value) if hasattr(value, "shape") else value,
        params,
    )


@dataclass(frozen=True)
class CheckpointBundle:
    """An immutable interpretation of a frozen Heat3D checkpoint."""

    path: Path
    sha256: str
    params: Any
    model_config: dict[str, Any]
    stats: dict[str, Any]
    epoch: int | None
    checkpoint_kind: str | None
    git_commit: str | None
    payload_metadata: dict[str, Any]

    def descriptor(self) -> dict[str, Any]:
        """Return JSON-safe checkpoint identity for equivalence records."""

        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "epoch": self.epoch,
            "checkpoint_kind": self.checkpoint_kind,
            "git_commit": self.git_commit,
            "model_config": _json_safe(self.model_config),
            "feature_names": list(self.stats.get("feature_names") or ()),
            "normalization_profile": self.stats.get("normalization_profile"),
            "condition_mean_shape": list(self.stats["condition_mean"].shape),
            "condition_std_shape": list(self.stats["condition_std"].shape),
        }


def load_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_epoch: int | None = None,
) -> CheckpointBundle:
    """Load and interpret one existing parameter checkpoint."""

    checkpoint_path = Path(path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")
    observed_sha256 = file_sha256(checkpoint_path)
    if expected_sha256 is not None and observed_sha256 != str(expected_sha256):
        raise ValueError(
            f"checkpoint SHA256 mismatch: expected={expected_sha256} "
            f"observed={observed_sha256}"
        )
    with checkpoint_path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dict")
    for field in ("params", "model_config", "train_only_normalization"):
        if field not in payload:
            raise ValueError(f"checkpoint missing {field}")
    epoch = None if payload.get("epoch") is None else int(payload["epoch"])
    if expected_epoch is not None and epoch != int(expected_epoch):
        raise ValueError(
            f"checkpoint epoch mismatch: expected={expected_epoch} observed={epoch}"
        )
    stats = materialize_checkpoint_stats(payload["train_only_normalization"])
    model_config = resolve_model_config(payload["model_config"], stats)
    metadata = {
        key: payload[key]
        for key in ("configuration_hash", "model_config_hash", "train_stats_hash", "param_count")
        if key in payload
    }
    return CheckpointBundle(
        path=checkpoint_path,
        sha256=observed_sha256,
        params=payload["params"],
        model_config=model_config,
        stats=stats,
        epoch=epoch,
        checkpoint_kind=(
            None if payload.get("checkpoint_kind") is None else str(payload["checkpoint_kind"])
        ),
        git_commit=(None if payload.get("git_commit") is None else str(payload["git_commit"])),
        payload_metadata=metadata,
    )


def load_run_config(path: str | Path) -> dict[str, Any]:
    """Load a persisted resolved run configuration as a pure mapping."""

    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"run config must be a JSON object: {config_path}")
    return config


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value
