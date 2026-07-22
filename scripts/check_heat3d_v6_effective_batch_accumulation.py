#!/usr/bin/env python3
"""Regression checks for the V6 graph-microbatch/effective-B28 contract."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_heat3d_v1_medium_controlled_training_export as runner


CONFIGS = (
    ROOT / "configs/heat3d_v6/V6_01_V4best.yaml",
    ROOT / "configs/heat3d_v6/V6_02_V5best.yaml",
)


def _group(count: int) -> dict[str, np.ndarray]:
    return {"target_normalized": np.zeros((count, 1, 1), dtype=np.float32)}


def _effective_windows() -> list[list[dict[str, np.ndarray]]]:
    counts: list[int] = []
    remaining = 768
    effective_remaining = 28
    while remaining:
        count = min(8, effective_remaining, remaining)
        counts.append(count)
        remaining -= count
        effective_remaining -= count
        if effective_remaining == 0:
            effective_remaining = 28
    return runner._gradient_accumulation_windows([_group(count) for count in counts], 28)


def _gradient_equivalence() -> dict[str, float]:
    x = jnp.arange(28 * 3, dtype=jnp.float32).reshape(28, 3) / 50.0
    y = jnp.sin(jnp.arange(28, dtype=jnp.float32) / 7.0)
    params = jnp.asarray([0.2, -0.1, 0.05], dtype=jnp.float32)

    def loss_fn(p, xb, yb):
        residual = xb @ p - yb
        return jnp.mean(jnp.square(residual))

    full_loss, full_grad = jax.value_and_grad(loss_fn)(params, x, y)
    starts = (0, 8, 16, 24)
    stops = (8, 16, 24, 28)
    weighted_loss = jnp.asarray(0.0)
    weighted_grad = jnp.zeros_like(params)
    for start, stop in zip(starts, stops, strict=True):
        micro_loss, micro_grad = jax.value_and_grad(loss_fn)(
            params, x[start:stop], y[start:stop]
        )
        count = stop - start
        weighted_loss += micro_loss * count
        weighted_grad += micro_grad * count
    accumulated_loss = weighted_loss / 28.0
    accumulated_grad = weighted_grad / 28.0

    clip_norm = 0.25

    def clip_once(grad):
        norm = jnp.linalg.norm(grad)
        return grad * jnp.minimum(1.0, clip_norm / jnp.maximum(norm, 1.0e-12))

    full_update = -1.0e-3 * clip_once(full_grad)
    accumulated_update = -1.0e-3 * clip_once(accumulated_grad)
    return {
        "loss_abs_error": float(jnp.abs(full_loss - accumulated_loss)),
        "gradient_max_abs_error": float(jnp.max(jnp.abs(full_grad - accumulated_grad))),
        "clipped_update_max_abs_error": float(
            jnp.max(jnp.abs(full_update - accumulated_update))
        ),
    }


def main() -> None:
    windows = _effective_windows()
    window_counts = [
        sum(runner._sample_count(group) for group in window) for window in windows
    ]
    micro_counts = [
        runner._sample_count(group) for window in windows for group in window
    ]
    assert window_counts == [28] * 27 + [12]
    assert sum(window_counts) == 768
    assert max(micro_counts) == 8
    assert len(windows) == 28

    numerical = _gradient_equivalence()
    assert numerical["loss_abs_error"] <= 1.0e-6
    assert numerical["gradient_max_abs_error"] <= 1.0e-6
    assert numerical["clipped_update_max_abs_error"] <= 1.0e-9

    config_contract = {}
    for path in CONFIGS:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        run = payload["overrides"]["run"]
        metadata = payload["overrides"]["metadata"]
        assert run["batch_size"] == 28
        assert run["micro_batch_size"] == 8
        assert run["drop_last"] is False
        assert metadata["optimizer_updates_per_epoch"] == 28
        assert metadata["final_partial_effective_batch_size"] == 12
        config_contract[path.stem] = {
            "configured_batch_size": 28,
            "effective_batch_size": 28,
            "graph_compatible_micro_batch_size": 8,
            "optimizer_updates_per_epoch": 28,
            "tail_effective_batch_size": 12,
        }

    print(
        json.dumps(
            {
                "status": "passed",
                "sample_count": 768,
                "optimizer_updates_per_epoch": len(windows),
                "window_sample_counts": window_counts,
                "micro_batch_count": len(micro_counts),
                "micro_batch_size_max": max(micro_counts),
                "tail_policy": "keep_sample_weighted_B12",
                "gradient_accumulation": "sample_count_weighted_mean",
                "gradient_clipping": "once_after_accumulation",
                "numerical_equivalence": numerical,
                "configs": config_contract,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
