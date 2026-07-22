#!/usr/bin/env python3
"""Regression checks for the V6 padded-geometry effective-B24 contract."""

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

from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder
from rigno.heat3d_v6_dataset import Heat3DV6DualRobinDataset
from scripts import run_heat3d_v1_medium_controlled_training_export as runner


CONFIGS = (
    ROOT / "configs/heat3d_v6/V6_01_V4best.yaml",
    ROOT / "configs/heat3d_v6/V6_02_V5best.yaml",
)
DATA_ROOT = ROOT / "data/heat3d_v6_p1g_geometry_deconfounded1024_v0"
MANIFEST = ROOT / "configs/heat3d_v6/v6_p1g_geometry_deconfounded1024_manifest.json"


def _group(count: int) -> dict[str, np.ndarray]:
    return {"target_normalized": np.zeros((count, 1, 1), dtype=np.float32)}


def _effective_windows() -> list[list[dict[str, np.ndarray]]]:
    micro_groups = [_group(8) for _ in range(96)]
    return runner._gradient_accumulation_windows(micro_groups, 24)


def _gradient_equivalence() -> dict[str, float]:
    x = jnp.arange(24 * 3, dtype=jnp.float32).reshape(24, 3) / 50.0
    y = jnp.sin(jnp.arange(24, dtype=jnp.float32) / 7.0)
    params = jnp.asarray([0.2, -0.1, 0.05], dtype=jnp.float32)

    def loss_fn(p, xb, yb):
        residual = xb @ p - yb
        return jnp.mean(jnp.square(residual))

    full_loss, full_grad = jax.value_and_grad(loss_fn)(params, x, y)
    weighted_loss = jnp.asarray(0.0)
    weighted_grad = jnp.zeros_like(params)
    for start in (0, 8, 16):
        micro_loss, micro_grad = jax.value_and_grad(loss_fn)(
            params, x[start : start + 8], y[start : start + 8]
        )
        weighted_loss += micro_loss * 8
        weighted_grad += micro_grad * 8
    accumulated_loss = weighted_loss / 24.0
    accumulated_grad = weighted_grad / 24.0

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


def _padding_contract() -> dict[str, object]:
    dataset = Heat3DV6DualRobinDataset(DATA_ROOT, MANIFEST)
    index = dataset.sample_index_by_id()
    selected = []
    groups = set()
    for sample_id in dataset.split_ids["train"]:
        example = dataset[index[sample_id]]
        group_id = str(example.meta["group_id"])
        if group_id in groups:
            continue
        selected.append(example)
        groups.add(group_id)
        if len(selected) == 8:
            break
    assert len(selected) == 8 and len(groups) == 8
    builder = Heat3DGraphBuilder(
        node_coordinate_encoding="raw",
        node_coordinate_freqs=4,
        radius_policy="discrete_physical_coverage",
        coverage_repair_policy="none",
        repair_p2r=True,
        repair_r2p=True,
        min_physical_coverage=1,
    )
    coords = [np.asarray(example.condition.coords) for example in selected]
    original = [
        builder.build_metadata(value, key=runner._metadata_key(0)) for value in coords
    ]
    padded, shared = runner._build_batch_metadata_with_seed(
        builder, coords, graph_seed=0
    )
    assert shared is False and len(padded) == 8
    edge_fields = (
        "p2r_edge_indices",
        "r2r_edge_indices",
        "r2r_edge_domains",
        "r2p_edge_indices",
    )
    target_lengths = {}
    for field in edge_fields:
        source_values = [getattr(metadata, field) for metadata in original]
        if all(value is None for value in source_values):
            assert getattr(padded, field) is None
            continue
        target = max(int(value.shape[1]) for value in source_values)
        target_lengths[field] = target
        packed = np.asarray(getattr(padded, field))
        assert packed.shape[0] == 8 and packed.shape[1] == target
        for row, source in enumerate(source_values):
            raw = np.asarray(source)[0]
            assert np.array_equal(packed[row, : raw.shape[0]], raw)
            if raw.shape[0] < target:
                expected = np.repeat(raw[-1:, :], target - raw.shape[0], axis=0)
                assert np.array_equal(packed[row, raw.shape[0] :], expected)
    graphs = builder.build_graphs(padded)
    assert all(
        np.all(np.isfinite(np.asarray(value)))
        for value in jax.tree_util.tree_leaves(graphs)
        if np.issubdtype(np.asarray(value).dtype, np.number)
    )
    return {
        "sample_count": 8,
        "distinct_geometry_groups": 8,
        "shared_metadata": shared,
        "real_edges_preserved_exactly": True,
        "padding_edges_are_repeated_dummy_edges": True,
        "padded_edge_lengths": target_lengths,
        "finite_graphs": True,
    }


def main() -> None:
    windows = _effective_windows()
    window_counts = [
        sum(runner._sample_count(group) for group in window) for window in windows
    ]
    micro_counts = [
        runner._sample_count(group) for window in windows for group in window
    ]
    assert window_counts == [24] * 32
    assert sum(window_counts) == 768
    assert micro_counts == [8] * 96
    assert len(windows) == 32
    assert all(len(window) == 3 for window in windows)

    numerical = _gradient_equivalence()
    assert numerical["loss_abs_error"] <= 1.0e-6
    assert numerical["gradient_max_abs_error"] <= 1.0e-6
    assert numerical["clipped_update_max_abs_error"] <= 1.0e-9
    padding = _padding_contract()

    config_contract = {}
    for path in CONFIGS:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        run = payload["overrides"]["run"]
        metadata = payload["overrides"]["metadata"]
        assert run["batch_size"] == 24
        assert run["micro_batch_size"] == 8
        assert run["drop_last"] is False
        assert metadata["micro_batches_per_epoch"] == 96
        assert metadata["optimizer_updates_per_epoch"] == 32
        assert metadata["final_partial_effective_batch_size"] is None
        assert metadata["cross_geometry_dummy_edge_padding"] is True
        config_contract[path.stem] = {
            "configured_batch_size": 24,
            "effective_batch_size": 24,
            "micro_batch_size": 8,
            "micro_batches_per_epoch": 96,
            "optimizer_updates_per_epoch": 32,
            "tail_effective_batch_size": None,
        }

    print(
        json.dumps(
            {
                "status": "passed",
                "sample_count": 768,
                "optimizer_updates_per_epoch": len(windows),
                "window_sample_counts": window_counts,
                "micro_batch_count": len(micro_counts),
                "micro_batch_sample_counts_unique": sorted(set(micro_counts)),
                "tail_policy": "none_exact_768_divisible_by_24_and_8",
                "gradient_accumulation": "3xB8_sample_count_weighted_mean",
                "gradient_clipping": "once_after_accumulation",
                "numerical_equivalence": numerical,
                "cross_geometry_padding": padding,
                "configs": config_contract,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
