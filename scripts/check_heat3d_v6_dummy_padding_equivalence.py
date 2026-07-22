#!/usr/bin/env python3
"""Check padded/unpadded V6 forward, loss, and gradient equivalence."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from check_heat3d_v4_registry import resolve_inherited_yaml  # noqa: E402
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.heat3d_v1_normalization import training_normalization_stats  # noqa: E402
from rigno.heat3d_v1_training_semantics import build_configured_zero_delta_bridge  # noqa: E402
from rigno.heat3d_v2_runner_command import build_training_command  # noqa: E402
from rigno.heat3d_v6_dataset import Heat3DV6DualRobinDataset  # noqa: E402
from rigno.models.rigno import RIGNO  # noqa: E402
import run_heat3d_v1_medium_controlled_training_export as runner  # noqa: E402


CONFIG = ROOT / "configs/heat3d_v6/V6_01_V4best_B32.yaml"


def _args(config: dict):
    values = list(build_training_command(config)[2:])
    wrapper_flags = {
        "--normalization-profile", "--condition-feature-transform",
        "--input-feature-schema", "--coord-policy", "--extent-feature-policy",
    }
    cleaned = []
    index = 0
    while index < len(values):
        if values[index] in wrapper_flags:
            index += 2
        else:
            cleaned.append(values[index])
            index += 1
    with patch.object(sys, "argv", ["padding-equivalence", *cleaned]):
        return runner.parse_args()


def _tree_max_abs(left, right) -> float:
    return max(
        float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
        for a, b in zip(
            jax.tree_util.tree_leaves(left),
            jax.tree_util.tree_leaves(right),
            strict=True,
        )
    )


def main() -> int:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config = resolve_inherited_yaml(raw, CONFIG)
    args = _args(config)
    dataset = Heat3DV6DualRobinDataset(
        ROOT / config["dataset"]["subset_path"],
        ROOT / config["dataset"]["manifest_path"],
    )
    index = dataset.sample_index_by_id()
    train = [dataset[index[sample_id]] for sample_id in dataset.split_ids["train"]]
    runner._bridge_for = lambda example: build_configured_zero_delta_bridge(
        example,
        input_feature_schema=config["dataset"]["input_feature_schema"],
        coord_policy=config["dataset"]["coord_policy"],
        extent_feature_policy=config["dataset"]["extent_feature_policy"],
    )
    stats = training_normalization_stats(
        train,
        normalization_profile=config["dataset"]["normalization_profile"],
        condition_feature_transform=config["dataset"]["condition_feature_transform"],
        input_feature_schema=config["dataset"]["input_feature_schema"],
        coord_policy=config["dataset"]["coord_policy"],
        extent_feature_policy=config["dataset"]["extent_feature_policy"],
    )
    builder = Heat3DGraphBuilder(**runner._graph_config_from_args(args))
    first = train[0]
    first_signature = runner._metadata_shape_signature(
        builder.build_metadata(first.condition.coords, key=runner._metadata_key(0))
    )
    second = next(
        example for example in train[1:]
        if runner._metadata_shape_signature(
            builder.build_metadata(example.condition.coords, key=runner._metadata_key(0))
        ) != first_signature
    )
    examples = [first, second]
    padded = runner._make_batch_group_with_seed(
        "padded_B2", examples, stats, builder, graph_seed=0
    )
    unpadded = [
        runner._make_batch_group_with_seed(
            f"unpadded_{position}", [example], stats, builder, graph_seed=0
        )
        for position, example in enumerate(examples)
    ]
    model_config = runner._resolve_decoder_bypass_model_config(
        runner._model_config_from_args(args), stats
    )
    model = RIGNO(**model_config)
    params = runner._model_init(
        model, jax.random.PRNGKey(0), padded, padded["inputs"]
    )["params"]
    padded_prediction = runner._model_apply(model, params, padded)
    unpadded_prediction = jnp.concatenate(
        [runner._model_apply(model, params, group) for group in unpadded], axis=0
    )
    loss_config = runner._loss_config_from_args(args)

    def padded_loss(current_params):
        return runner._loss_components(
            model, current_params, [padded], stats, loss_config
        )["total_loss"]

    def unpadded_loss(current_params):
        return runner._loss_components(
            model, current_params, unpadded, stats, loss_config
        )["total_loss"]

    loss_padded, grad_padded = jax.value_and_grad(padded_loss)(params)
    loss_unpadded, grad_unpadded = jax.value_and_grad(unpadded_loss)(params)
    jax.block_until_ready(loss_padded)
    report = {
        "status": "passed",
        "config_id": raw["config_id"],
        "sample_ids": [example.sample_id for example in examples],
        "distinct_geometry_groups": len({example.meta["group_id"] for example in examples}),
        "forward_max_abs_error": float(
            np.max(np.abs(np.asarray(padded_prediction) - np.asarray(unpadded_prediction)))
        ),
        "loss_abs_error": abs(float(loss_padded) - float(loss_unpadded)),
        "gradient_max_abs_error": _tree_max_abs(grad_padded, grad_unpadded),
        "real_graph_semantics": "unchanged; padding repeats dummy-to-dummy edges only",
        "target_or_label_used_for_padding": False,
    }
    tolerances = {"forward": 2e-5, "loss": 2e-6, "gradient": 2e-5}
    report["tolerances"] = tolerances
    assert report["distinct_geometry_groups"] == 2
    assert report["forward_max_abs_error"] <= tolerances["forward"]
    assert report["loss_abs_error"] <= tolerances["loss"]
    assert report["gradient_max_abs_error"] <= tolerances["gradient"]
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
