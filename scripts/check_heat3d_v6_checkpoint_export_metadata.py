#!/usr/bin/env python3
"""Regression check for checkpoint metadata export with graph builders."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_heat3d_v1_medium_controlled_training_export as runner  # noqa: E402


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        dataset_loader="v6_p1i_dual_robin_manifest_v1",
        dataset_manifest=Path("manifest.json"),
        split_map=Path("split.json"),
        epochs=600,
        output_dir=Path("output/run"),
        boundary_mask_fallback=False,
        prediction_split="valid_iid",
        init_checkpoint=None,
        checkpoint_load_strict=None,
        partial_load_policy=None,
    )


def _metadata(builder):
    return runner._checkpoint_run_metadata(
        sample_root=Path("data/subset"),
        args=_args(),
        split_source="frozen_manifest",
        split_counts={"train": 768, "valid_iid": 128, "test_iid": 128},
        model_config={},
        loss_config={},
        lr_config={},
        optimizer_config={},
        seed_config={},
        batch_config={},
        graph_config={},
        builder=builder,
    )


def main() -> int:
    without_cache = _metadata(None)
    if without_cache["run_shared_support_graph_cache"] is not None:
        raise AssertionError("non-shared builder must record a null cache audit")

    shared = runner.RunSharedSupportGraphBuilder(builder=object())
    shared.audit["requested_metadata_calls"] = 7
    with_cache = _metadata(shared)
    if with_cache["run_shared_support_graph_cache"] != dict(shared.audit):
        raise AssertionError("shared builder cache audit was not exported")
    if list(with_cache).count("run_shared_support_graph_cache") != 1:
        raise AssertionError("cache audit metadata key must be unique")

    print("checkpoint export metadata regression: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
