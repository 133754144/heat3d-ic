#!/usr/bin/env python3
"""Check V4 decoder bypass selector and zero-residual initialization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import jax
import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (REPO_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_heat3d_v1_medium_controlled_training_export as runner  # noqa: E402
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.heat3d_v1_native_supervised import Heat3DV1NativeSupervisedDataset  # noqa: E402
from rigno.models.rigno import RIGNO  # noqa: E402


DEFAULT_SUBSET = (
    REPO_DIR
    / "data"
    / "heat3d-thermal-simulation"
    / "subsets"
    / "v1_multilayer_bc_eq_supervised_small"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--max-examples", type=int, default=2)
    parser.add_argument("--atol", type=float, default=1.0e-8)
    return parser.parse_args()


def _sample_root(path: Path) -> Path:
    return path / "samples" if (path / "samples").exists() else path


def _max_abs(value: Any) -> float:
    array = np.asarray(value)
    return float(np.max(np.abs(array))) if array.size else 0.0


def main() -> int:
    args = parse_args()
    sample_root = _sample_root(args.subset).resolve()
    dataset = Heat3DV1NativeSupervisedDataset(sample_root, k_encoding_mode="diag3")
    examples = sorted(dataset.samples, key=lambda item: item.sample_id)
    examples = examples[: max(1, args.max_examples)]
    if not examples:
        raise ValueError(f"No usable examples found in {sample_root}")

    stats = runner._train_only_stats(examples)
    model_config = dict(runner.RUNNER_MODEL_CONFIG)
    model_config.update(
        {
            "node_latent_size": 8,
            "edge_latent_size": 8,
            "processor_steps": 1,
            "mlp_hidden_layers": 1,
            "p_edge_masking": 0.0,
            "decoder_bypass_mode": runner.DECODER_BYPASS_MODE_POST_DECODER_RESIDUAL,
            "decoder_bypass_features": runner.DECODER_BYPASS_FEATURES_FULL_CONDITION,
            "decoder_bypass_feature_source": runner.DECODER_BYPASS_FEATURE_SOURCE_NORMALIZED_C,
            "decoder_bypass_hidden_size": 4,
            "decoder_bypass_layers": 1,
            "decoder_bypass_init": runner.DECODER_BYPASS_INIT_ZERO_RESIDUAL,
            "decoder_bypass_residual_scale": 1.0,
        }
    )
    model_config = runner._resolve_decoder_bypass_model_config(model_config, stats)

    builder = Heat3DGraphBuilder(
        radius_policy="discrete_physical_coverage",
        coverage_repair_policy="none",
        repair_p2r=True,
        repair_r2p=True,
        min_physical_coverage=1,
    )
    group = runner._make_groups_with_progress(
        examples,
        stats,
        builder,
        "decoder_bypass_check",
        False,
        "basic",
        0,
        batch_size=len(examples),
        drop_last=False,
    )[0]
    runner._check_decoder_bypass_input_alignment(model_config, [group])

    model_scale1 = RIGNO(**model_config)
    variables = model_scale1.init(
        jax.random.PRNGKey(0),
        inputs=group["inputs"],
        graphs=group["graphs"],
    )
    output_scale1 = model_scale1.apply(
        variables,
        inputs=group["inputs"],
        graphs=group["graphs"],
    )
    model_scale0_config = dict(model_config)
    model_scale0_config["decoder_bypass_residual_scale"] = 0.0
    output_scale0 = RIGNO(**model_scale0_config).apply(
        variables,
        inputs=group["inputs"],
        graphs=group["graphs"],
    )
    output_diff = _max_abs(np.asarray(output_scale1) - np.asarray(output_scale0))

    payload = {
        "script": Path(__file__).name,
        "non_execution": "no training, no evaluation, no artifact writes",
        "subset": str(sample_root),
        "sample_ids": [example.sample_id for example in examples],
        "decoder_bypass_feature_names": list(model_config["decoder_bypass_feature_names"]),
        "decoder_bypass_feature_indices": list(model_config["decoder_bypass_feature_indices"]),
        "decoder_bypass_num_features": model_config["decoder_bypass_num_features"],
        "decoder_bypass_output_space": model_config["decoder_bypass_output_space"],
        "zero_residual_output_max_abs_diff": output_diff,
        "passed": bool(output_diff <= args.atol),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
