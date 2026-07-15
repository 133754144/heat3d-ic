#!/usr/bin/env python3
"""Short train-only scale-head probes over cached N3 frozen features.

No RIGNO/GNN object is constructed here.  Each probe reads cached N3 feature
arrays, optimizes a small scale head on train only, and reports valid_iid
metrics without touching test, hard, or sealed-IID roles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any

from flax import linen as nn
import jax
import jax.numpy as jnp
import jax.tree_util as tree
import numpy as np
import optax


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rigno.heat3d_v5_scale_pooling import QK_REGION_FEATURES  # noqa: E402


PROBE_SPECS = (
    {"probe_id": "mean", "pooling": "mean", "depth": 1},
    {"probe_id": "mean_plus_std", "pooling": "mean_std", "depth": 1},
    {"probe_id": "mean_plus_max", "pooling": "mean_max", "depth": 1},
    {"probe_id": "pre_film_mean_plus_std", "pooling": "pre_film_mean_std", "depth": 1},
    {"probe_id": "deep_scale_head", "pooling": "mean", "depth": 3},
    {"probe_id": "latent_attention_pooling", "pooling": "latent_attention", "depth": 1},
    {"probe_id": "qk_gated_pooling", "pooling": "qk_gated", "depth": 1},
)
EPS = 1.0e-12


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--seed", type=int, default=2026071601)
    return parser.parse_args()


class FrozenScaleProbe(nn.Module):
    """Small identity-initialized residual scale head for one frozen feature mode."""

    pooling: str
    depth: int
    hidden_size: int = 64

    @nn.compact
    def __call__(
        self,
        global_context: jnp.ndarray,
        rnodes_processed: jnp.ndarray,
        rnodes_processed_pre_film: jnp.ndarray,
        qk_region_features: jnp.ndarray,
    ) -> jnp.ndarray:
        post = rnodes_processed
        pre = rnodes_processed_pre_film
        if self.pooling == "mean":
            pooled = jnp.mean(post, axis=1)
        elif self.pooling == "mean_std":
            pooled = jnp.concatenate([jnp.mean(post, axis=1), jnp.std(post, axis=1)], axis=-1)
        elif self.pooling == "mean_max":
            pooled = jnp.concatenate([jnp.mean(post, axis=1), jnp.max(post, axis=1)], axis=-1)
        elif self.pooling == "pre_film_mean_std":
            pooled = jnp.concatenate([jnp.mean(pre, axis=1), jnp.std(pre, axis=1)], axis=-1)
        elif self.pooling == "latent_attention":
            hidden = nn.gelu(nn.Dense(self.hidden_size, name="latent_attention_hidden")(post))
            logits = nn.Dense(
                1,
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.zeros,
                name="latent_attention_logits",
            )(hidden)[..., 0]
            weights = jax.nn.softmax(logits, axis=1)
            pooled = jnp.sum(weights[..., None] * post, axis=1)
        elif self.pooling == "qk_gated":
            mean_pool = jnp.mean(post, axis=1)
            hidden = nn.gelu(nn.Dense(self.hidden_size, name="qk_attention_hidden")(qk_region_features))
            logits = nn.Dense(
                1,
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.zeros,
                name="qk_attention_logits",
            )(hidden)[..., 0]
            weights = jax.nn.softmax(logits, axis=1)
            attention_pool = jnp.sum(weights[..., None] * post, axis=1)
            residual_input = jnp.concatenate(
                [attention_pool - mean_pool, mean_pool, jnp.mean(qk_region_features, axis=1)],
                axis=-1,
            )
            residual = nn.Dense(
                post.shape[-1],
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.zeros,
                name="qk_attention_residual",
            )(residual_input)
            pooled = mean_pool + residual
        else:  # pragma: no cover - fixed spec list guards this.
            raise ValueError(f"unsupported frozen probe pooling {self.pooling!r}")
        features = jnp.concatenate([global_context, pooled], axis=-1)
        hidden = nn.gelu(nn.Dense(self.hidden_size, name="global_scale_hidden")(features))
        for index in range(self.depth - 1):
            hidden = nn.gelu(
                nn.Dense(self.hidden_size, name=f"global_scale_extra_hidden_{index}")(hidden)
            )
        return nn.Dense(
            1,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="global_scale_output",
        )(hidden)[:, 0]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _device_peak_mb() -> float | None:
    values = []
    for device in jax.devices():
        stats = device.memory_stats()
        if not stats:
            continue
        for key in ("peak_bytes_in_use", "peak_pool_bytes", "bytes_in_use"):
            value = stats.get(key)
            if value is not None:
                values.append(float(value) / (1024.0 * 1024.0))
    return max(values) if values else None


def _parameter_count(params: Any) -> int:
    return int(sum(np.asarray(value).size for value in tree.tree_leaves(params)))


def _validate_cache(manifest: dict[str, Any], cache_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    if manifest.get("config_id") != "V4P5_07_native_pooled_latent_global_film":
        raise ValueError("frozen probe cache is not tied to N3")
    if manifest.get("checkpoint_kind") != "best" or int(manifest.get("checkpoint_epoch", -1)) != 402:
        raise ValueError("frozen probe cache must be tied to N3 best e402")
    if manifest.get("roles_materialized") != ["train", "valid_iid"]:
        raise ValueError("frozen probe cache role contract drifted")
    if manifest.get("forbidden_roles_materialized") != [] or manifest.get("sealed_iid_accessed"):
        raise ValueError("frozen probe cache touched forbidden roles")
    standardizer = manifest.get("global_context_standardizer") or {}
    if standardizer.get("fit_population") != "train_only":
        raise ValueError("global context standardizer was not fit on train only")
    if tuple(manifest.get("qk_region_feature_names") or ()) != QK_REGION_FEATURES:
        raise ValueError("qk regional feature schema drifted")
    arrays = {}
    for split in ("train", "valid_iid"):
        record = manifest["splits"][split]
        path = Path(record["artifact"])
        if not path.is_absolute():
            path = cache_dir / path.name
        if not path.is_file() or _sha256(path) != record["artifact_sha256"]:
            raise ValueError(f"{split}: frozen feature artifact SHA mismatch")
        arrays[split] = _load_cache(path)
    required = {
        "global_context", "rnodes_processed", "rnodes_processed_pre_film", "qk_region_features",
        "phi_hat", "s_phys", "s_true", "target_deltaT", "control_volumes",
        "dirichlet_mask", "dirichlet_delta",
    }
    for split, payload in arrays.items():
        missing = required - set(payload)
        if missing:
            raise ValueError(f"{split}: frozen cache fields missing: {sorted(missing)}")
        if not all(np.all(np.isfinite(value)) for name, value in payload.items() if name != "sample_ids"):
            raise ValueError(f"{split}: frozen cache contains non-finite values")
    return arrays["train"], arrays["valid_iid"]


def _as_inputs(cache: dict[str, np.ndarray]) -> dict[str, jnp.ndarray]:
    return {
        name: jnp.asarray(cache[name], dtype=jnp.float32)
        for name in (
            "global_context", "rnodes_processed", "rnodes_processed_pre_film", "qk_region_features"
        )
    }


def _scale_targets(cache: dict[str, np.ndarray]) -> jnp.ndarray:
    return jnp.log(jnp.maximum(jnp.asarray(cache["s_true"]), EPS)) - jnp.log(
        jnp.maximum(jnp.asarray(cache["s_phys"]), EPS)
    )


def _field_metrics(residual_scale: np.ndarray, cache: dict[str, np.ndarray]) -> dict[str, float]:
    s_hat = np.asarray(cache["s_phys"], dtype=np.float64) * np.exp(np.asarray(residual_scale, dtype=np.float64))
    phi_hat = np.asarray(cache["phi_hat"], dtype=np.float64)
    prediction = s_hat[:, None] * phi_hat
    mask = np.asarray(cache["dirichlet_mask"], dtype=np.float64)
    prediction = np.where(mask > 0.5, np.asarray(cache["dirichlet_delta"], dtype=np.float64), prediction)
    target = np.asarray(cache["target_deltaT"], dtype=np.float64)
    weights = np.asarray(cache["control_volumes"], dtype=np.float64)
    error = prediction - target
    per_sample_cv_mse = np.sum(np.square(error) * weights, axis=1) / np.maximum(
        np.sum(weights, axis=1), EPS
    )
    s_true = np.asarray(cache["s_true"], dtype=np.float64)
    sample_relative = np.sqrt(per_sample_cv_mse) / np.maximum(s_true, EPS)
    global_relative = np.sqrt(np.sum(np.square(error) * weights) / np.maximum(
        np.sum(np.square(target) * weights), EPS
    ))
    scale_log_error = np.log(np.maximum(s_hat, EPS)) - np.log(np.maximum(s_true, EPS))
    return {
        "scale_log_rmse": float(np.sqrt(np.mean(np.square(scale_log_error)))),
        "fixed_shape_joint_point_global_relative_rmse_pct": float(100.0 * global_relative),
        "fixed_shape_joint_sample_first_cv_relative_rmse_pct": float(100.0 * np.mean(sample_relative)),
        "fixed_shape_joint_raw_cv_weighted_rmse_K": float(np.sqrt(np.mean(per_sample_cv_mse))),
        "fixed_shape_joint_point_sse_K2": float(np.sum(np.square(error))),
    }


def _run_probe(
    spec: dict[str, Any],
    train_cache: dict[str, np.ndarray],
    valid_cache: dict[str, np.ndarray],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> dict[str, Any]:
    train_inputs = _as_inputs(train_cache)
    valid_inputs = _as_inputs(valid_cache)
    train_targets = _scale_targets(train_cache)
    valid_targets = _scale_targets(valid_cache)
    model = FrozenScaleProbe(pooling=str(spec["pooling"]), depth=int(spec["depth"]))
    key = jax.random.PRNGKey(seed)
    params = model.init(key, **{name: value[:1] for name, value in train_inputs.items()})["params"]
    tx = optax.adamw(learning_rate=float(lr), weight_decay=1.0e-4)
    state = tx.init(params)

    @jax.jit
    def train_step(current_params, current_state, inputs, target):
        def loss_fn(candidate):
            prediction = model.apply({"params": candidate}, **inputs)
            return jnp.mean(jnp.square(prediction - target))
        loss, grads = jax.value_and_grad(loss_fn)(current_params)
        updates, next_state = tx.update(grads, current_state, current_params)
        next_params = optax.apply_updates(current_params, updates)
        return next_params, next_state, loss

    @jax.jit
    def infer(current_params, inputs):
        return model.apply({"params": current_params}, **inputs)

    started = time.perf_counter()
    initial_valid_residual = np.asarray(infer(params, valid_inputs))
    initial_metrics = _field_metrics(initial_valid_residual, valid_cache)
    rng = np.random.default_rng(seed)
    count = int(train_targets.shape[0])
    for _ in range(int(epochs)):
        permutation = rng.permutation(count)
        for start in range(0, count, int(batch_size)):
            selection = permutation[start : start + int(batch_size)]
            batch = {name: value[selection] for name, value in train_inputs.items()}
            params, state, _ = train_step(params, state, batch, train_targets[selection])
    residual = np.asarray(infer(params, valid_inputs))
    runtime_s = time.perf_counter() - started
    metrics = _field_metrics(residual, valid_cache)
    result = {
        "probe_id": spec["probe_id"],
        "scale_pooling": spec["pooling"],
        "scale_head_depth": int(spec["depth"]),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(lr),
        "seed": int(seed),
        "parameter_count": _parameter_count(params),
        "peak_rss_mb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0,
        "peak_device_memory_mb": _device_peak_mb(),
        "runtime_s": float(runtime_s),
        "initial_valid": initial_metrics,
        "valid": metrics,
        "finite": bool(np.all(np.isfinite(residual))),
        "gnn_backward": False,
        "training_started": True,
    }
    return result


def main() -> int:
    args = _parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0.0:
        raise ValueError("epochs/batch-size/lr must be positive")
    cache_dir = args.cache_dir.resolve()
    manifest_path = cache_dir / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_cache, valid_cache = _validate_cache(manifest, cache_dir)
    started = time.perf_counter()
    results = [
        _run_probe(
            spec,
            train_cache,
            valid_cache,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed + index,
        )
        for index, spec in enumerate(PROBE_SPECS)
    ]
    ranked = sorted(
        results,
        key=lambda item: (
            float(item["valid"]["scale_log_rmse"]),
            float(item["valid"]["fixed_shape_joint_point_global_relative_rmse_pct"]),
        ),
    )
    for rank, item in enumerate(ranked, start=1):
        item["rank_by_valid_scale_log_rmse"] = int(rank)
    payload = {
        "schema_version": "heat3d_v5_gate6f_frozen_scale_probe_v1",
        "cache_manifest": str(manifest_path),
        "cache_manifest_sha256": _sha256(manifest_path),
        "config_id": manifest["config_id"],
        "checkpoint_epoch": manifest["checkpoint_epoch"],
        "roles_accessed": ["train", "valid_iid"],
        "forbidden_roles_accessed": [],
        "sealed_iid_accessed": False,
        "model_inference_run": False,
        "gnn_backward": False,
        "frozen_probe_short_training": True,
        "xla_python_client_preallocate": os.environ.get(
            "XLA_PYTHON_CLIENT_PREALLOCATE", "unset"
        ),
        "probe_input_fields": [
            "global_context_train_standardized", "rnodes_processed", "rnodes_processed_pre_film",
            "qk_region_features_raw_coords_k_q_bc", "s_phys",
        ],
        "target_fields_used_only_for_scale_head_supervision_and_valid_reporting": [
            "s_true", "target_deltaT", "control_volumes", "dirichlet_mask", "dirichlet_delta",
        ],
        "qk_region_feature_names": list(QK_REGION_FEATURES),
        "results": results,
        "ranking": [item["probe_id"] for item in ranked],
        "elapsed_s": float(time.perf_counter() - started),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
