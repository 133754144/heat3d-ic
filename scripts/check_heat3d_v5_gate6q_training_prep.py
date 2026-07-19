#!/usr/bin/env python3
"""Validate Gate 6Q V42/V43/V44 contracts without starting training."""

from __future__ import annotations

import argparse
import copy
import csv
import inspect
import json
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace
from typing import Any

import jax.numpy as jnp
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rigno.heat3d_v2_config import validate_v2_config  # noqa: E402
from rigno.heat3d_v2_runner_command import build_training_command  # noqa: E402
from rigno.heat3d_v5_metrics import control_volume_weights  # noqa: E402
from rigno.heat3d_v5_scale_context import (  # noqa: E402
    XY_SCALE_CONTEXT_FEATURES,
    fit_train_only_scale_context_standardizer,
    p2r_partition_of_unity_audit,
    regional_source_volume_weights_from_raw,
    standardize_scale_contexts,
    xy_scale_context_from_raw_condition,
)
from rigno.heat3d_v5_shape_scale import native_shape_scale_losses  # noqa: E402
from rigno.models.rigno import RIGNO  # noqa: E402
from scripts.check_heat3d_v4_registry import resolve_inherited_yaml  # noqa: E402
from scripts.run_heat3d_v1_medium_controlled_training_export import (  # noqa: E402
    _fit_native_loss_train_references,
)
from scripts.smoke_heat3d_v5_gate6q_single_batch import run_smoke  # noqa: E402


BASELINE = ROOT / "configs/heat3d_v5/generated/V4P5_38_gate6n_v36_r2r_mask_p005_e600.yaml"
CONFIGS = {
    "V42": ROOT / "configs/heat3d_v5/generated/V4P5_42_gate6q_objective_only_e600.yaml",
    "V43": ROOT / "configs/heat3d_v5/generated/V4P5_43_gate6q_xy_scale_features_e600.yaml",
    "V44": ROOT / "configs/heat3d_v5/generated/V4P5_44_gate6q_xy_deepsets_e600.yaml",
}
REGISTRY = ROOT / "configs/heat3d_v5/v5_gate6q_training_registry.csv"
REPORT = ROOT / "configs/heat3d_v5/gate6q_training/gate6q_training_prep.json"
FORBIDDEN = {
    "test_iid",
    "hard_train_holdout",
    "hard_challenge_valid",
    "hard_challenge_test",
    "sealed_iid",
}
IDENTITY_FIELDS = {
    "config_id",
    "description",
    "export.output_dir",
    "export.run_name",
    "run.final_probe_output_dir",
    "run.post_training_diagnostics_output_dir",
}


def _resolved(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    resolved = resolve_inherited_yaml(payload, path)
    resolved["config_id"] = payload["config_id"]
    validate_v2_config(resolved, config_path=path)
    return resolved


def _semantic_defaults(config: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    result["model"].setdefault("scale_context_mode", "none")
    result["model"].setdefault("scale_context_feature_names", [])
    result["model"].setdefault("scale_deepsets_mode", "none")
    result["model"].setdefault("scale_deepsets_hidden_size", 64)
    result["loss"].setdefault("native_raw_loss_mode", "per_sample_cv_mse")
    result["loss"].setdefault("native_log_scale_weight_mode", "uniform")
    result["loss"].setdefault("native_log_scale_weight_clip_min", 0.25)
    result["loss"].setdefault("native_log_scale_weight_clip_max", 4.0)
    return result


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(child, path))
        return result
    return {prefix: value}


def _science(config: dict[str, Any]) -> dict[str, Any]:
    result = _semantic_defaults(config)
    for key in ("schema_version", "config_id", "description", "metadata"):
        result.pop(key, None)
    for field in ("final_probe_output_dir", "post_training_diagnostics_output_dir"):
        result["run"].pop(field, None)
    for field in ("output_dir", "run_name"):
        result["export"].pop(field, None)
    return result


def _diff(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    flat_left = _flatten(_science(left))
    flat_right = _flatten(_science(right))
    return {
        key: {"left": flat_left.get(key), "right": flat_right.get(key)}
        for key in sorted(set(flat_left) | set(flat_right))
        if flat_left.get(key) != flat_right.get(key)
    }


def _parameter_paths(params: Any) -> dict[str, Any]:
    import jax

    return {
        "/".join(
            str(getattr(item, "key", getattr(item, "name", item)))
            for item in path
        ): value
        for path, value in jax.tree_util.tree_flatten_with_path(params)[0]
    }


def _objective_and_feature_fixtures() -> None:
    target = jnp.asarray(
        [[[[0.0], [1.0], [2.0]]], [[[0.0], [2.0], [4.0]]]],
        dtype=jnp.float32,
    )
    prediction_field = target + jnp.asarray(
        [[[[0.0], [0.5], [-0.5]]], [[[0.0], [1.0], [-1.0]]]],
        dtype=jnp.float32,
    )
    volumes = jnp.ones((2, 3), dtype=jnp.float32)
    mask = jnp.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    scale = jnp.sqrt(jnp.mean(jnp.square(target[:, 0, :, 0]), axis=1))
    losses = native_shape_scale_losses(
        {
            "phi_hat": target / scale[:, None, None, None],
            "s_hat": scale[:, None, None, None] * 1.1,
            "deltaT_hat": prediction_field,
        },
        target_deltaT=target,
        control_volumes=volumes,
        dirichlet_mask=mask,
        loss_weights={
            "shape_cv": 1.5,
            "log_scale": 0.5,
            "relative_field": 1.0,
            "raw_absolute": 1.0,
        },
        raw_loss_mode="point_global_fixed_train_energy_sse",
        raw_train_target_energy_per_point=float(jnp.mean(jnp.square(target))),
        log_scale_weight_mode="train_true_scale_squared_clipped",
        log_scale_train_true_scale_sq_mean=float(jnp.mean(jnp.square(scale))),
        log_scale_weight_clip=(0.25, 4.0),
    )
    manual = float(
        jnp.sum(jnp.square(prediction_field - target))
        / jnp.sum(jnp.square(target))
    )
    assert np.isclose(float(losses["raw_absolute_field_loss"]), manual)

    coords = np.stack(
        np.meshgrid(
            np.asarray([0.0, 1.0]),
            np.asarray([0.0, 1.0]),
            np.asarray([0.0, 1.0]),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    fit_mask = (coords[:, 2] == 0.0).astype(np.float32)
    fit_targets = np.stack(
        [
            np.linspace(0.0, 2.0, coords.shape[0]),
            np.linspace(0.0, 4.0, coords.shape[0]),
        ]
    )[:, None, :, None]
    config = {
        "native_raw_loss_mode": "point_global_fixed_train_energy_sse",
        "native_log_scale_weight_mode": "train_true_scale_squared_clipped",
        "native_log_scale_weight_clip_min": 0.25,
        "native_log_scale_weight_clip_max": 4.0,
    }
    examples = []
    for index in range(fit_targets.shape[0]):
        target_row = fit_targets[index : index + 1]
        relative = SimpleNamespace(
            condition_feature_names=("is_bottom",),
            condition_features=fit_mask[:, None],
        )
        example = SimpleNamespace(
            sample_id=f"train_{index}",
            condition=SimpleNamespace(coords=coords),
            get_relative_bc_feature_view=lambda relative=relative: relative,
            build_temperature_rise_legacy_inputs_from_relative_features=(
                lambda bridge_policy, target_row=target_row: SimpleNamespace(
                    target_delta_u=target_row
                )
            ),
        )
        examples.append(example)
    _fit_native_loss_train_references(config, examples)
    cv = np.asarray(control_volume_weights(coords))
    expected_reference = float(
        np.mean(
            [
                np.sum(
                    np.square(fit_targets[index, 0, :, 0] * (1.0 - fit_mask))
                    * cv
                )
                / np.sum(cv)
                for index in range(fit_targets.shape[0])
            ]
        )
    )
    assert np.isclose(
        config["native_log_scale_train_true_scale_sq_mean"],
        expected_reference,
    )
    assert np.isclose(
        config["native_raw_train_target_energy_per_point"],
        float(np.mean(np.square(fit_targets))),
    )
    weight_diagnostics = config["native_log_scale_weight_diagnostics"]
    assert weight_diagnostics["fit_roles"] == ["train"]
    assert weight_diagnostics["sample_count"] == 2
    assert 0.0 < weight_diagnostics["effective_sample_size"] <= 2.0

    raw = np.stack(
        [
            np.linspace(1.0, 4.0, coords.shape[0]),
            np.linspace(4.0, 1.0, coords.shape[0]),
            np.zeros(coords.shape[0]),
        ],
        axis=1,
    )
    row = xy_scale_context_from_raw_condition(
        coords=coords,
        raw_condition=raw,
        condition_feature_names=("k_x", "k_y", "q"),
    )
    assert tuple(row) == XY_SCALE_CONTEXT_FEATURES
    assert all(np.isfinite(value) for value in row.values())
    assert all(
        row[name] == 0.0
        for name in XY_SCALE_CONTEXT_FEATURES
        if name.startswith("source_") or name.startswith("q_")
    )
    standardizer = fit_train_only_scale_context_standardizer(
        [row, row], fit_sample_ids=["train_a", "train_b"]
    )
    encoded = standardize_scale_contexts([row], standardizer)
    assert encoded.shape == (1, len(XY_SCALE_CONTEXT_FEATURES))
    assert standardizer["fit_roles"] == ["train"]

    condition_names = ("q",)
    condition = np.linspace(0.0, 2.0, coords.shape[0])[:, None]
    edges = []
    for physical in range(coords.shape[0]):
        edges.append((physical, physical % 3))
        if physical % 2 == 0:
            edges.append((physical, (physical + 1) % 3))
        if physical % 4 == 0:
            edges.append((physical, (physical + 2) % 3))
    edges.append((-1, -1))
    edge_array = np.asarray(edges, dtype=np.int64)
    source, regional_volume = regional_source_volume_weights_from_raw(
        coords=coords,
        raw_condition=condition,
        condition_feature_names=condition_names,
        p2r_edge_indices=edge_array,
        rnode_count=3,
    )
    partition = p2r_partition_of_unity_audit(
        coords=coords,
        raw_condition=condition,
        condition_feature_names=condition_names,
        p2r_edge_indices=edge_array,
        rnode_count=3,
    )
    assert partition["zero_degree_node_count"] == 0
    assert partition["maximum_degree"] > partition["minimum_degree"]
    assert partition["maximum_partition_of_unity_error"] <= 1.0e-12
    assert partition["source_conserved"]
    assert partition["volume_conserved"]
    assert np.isclose(np.sum(source), partition["physical_source_total"])
    assert np.isclose(
        np.sum(regional_volume), partition["physical_volume_total"]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-runtime", action="store_true")
    args = parser.parse_args()
    _objective_and_feature_fixtures()

    baseline = _resolved(BASELINE)
    configs = {name: _resolved(path) for name, path in CONFIGS.items()}
    diffs = {
        "V42_vs_V38": _diff(baseline, configs["V42"]),
        "V43_vs_V38": _diff(baseline, configs["V43"]),
        "V44_vs_V43": _diff(configs["V43"], configs["V44"]),
    }
    assert set(diffs["V42_vs_V38"]) == {
        "loss.native_log_scale_weight_mode",
        "loss.native_raw_loss_mode",
    }
    assert set(diffs["V43_vs_V38"]) == {
        "model.scale_context_feature_names",
        "model.scale_context_mode",
    }
    assert set(diffs["V44_vs_V43"]) == {"model.scale_deepsets_mode"}

    invariant_paths = (
        "dataset",
        "graph",
        "optimizer",
        "run.epochs",
        "run.batch_size",
        "run.validation_batch_size",
        "run.prediction_batch_size",
        "run.batch_plan",
        "run.batch_build_seed",
        "model.node_latent_size",
        "model.edge_latent_size",
        "model.processor_steps",
        "model.mlp_hidden_layers",
        "model.p_edge_masking",
        "model.edge_masking_scope",
    )
    flat_base = _flatten(baseline)
    for candidate in configs.values():
        flat_candidate = _flatten(candidate)
        for path in invariant_paths:
            if path in {"dataset", "graph", "optimizer"}:
                assert candidate[path] == baseline[path]
            else:
                assert flat_candidate[path] == flat_base[path]
        assert candidate["run"]["init_checkpoint"] is None
        assert candidate["run"]["epochs"] == 600
        assert candidate["model"]["p_edge_masking"] == 0.05
        assert candidate["model"]["edge_masking_scope"] == "r2r_only"
        assert candidate["export"]["prediction_split"] == "valid_iid"
        assert set(candidate["metadata"]["forbidden_access_roles"]) == FORBIDDEN
        assert candidate["metadata"]["training_started"] is False
        assert candidate["metadata"]["formal_e600_started"] is False
        command = build_training_command(candidate, python_executable="python")
        command_text = shlex.join(command)
        assert "--epochs 600" in command_text
        assert "--init-checkpoint" not in command_text
        assert "--prediction-split valid_iid" in command_text

    assert configs["V42"]["loss"]["native_shape_cv_weight"] == baseline["loss"]["native_shape_cv_weight"]
    assert configs["V42"]["loss"]["native_log_scale_weight"] == baseline["loss"]["native_log_scale_weight"]
    assert configs["V42"]["loss"]["native_relative_field_weight"] == baseline["loss"]["native_relative_field_weight"]
    assert configs["V42"]["loss"]["native_raw_field_weight"] == baseline["loss"]["native_raw_field_weight"]
    assert configs["V43"]["loss"] == baseline["loss"]
    assert configs["V44"]["loss"] == configs["V43"]["loss"]
    assert tuple(configs["V43"]["model"]["scale_context_feature_names"]) == XY_SCALE_CONTEXT_FEATURES
    assert configs["V44"]["model"]["scale_deepsets_mode"] == "source_volume_residual"
    assert "source/volume-aware latent DeepSets" in CONFIGS["V44"].read_text(
        encoding="utf-8"
    )
    assert "does not claim explicit regional XY physics" in CONFIGS[
        "V44"
    ].read_text(encoding="utf-8")

    feature_signature = set(
        inspect.signature(xy_scale_context_from_raw_condition).parameters
    )
    assert not feature_signature.intersection(
        {"target", "target_deltaT", "temperature", "label"}
    )
    prediction_source = inspect.getsource(RIGNO.predict_native_shape_scale)
    assert prediction_source.index("phi_hat =") < prediction_source.index(
        "scale_context_array ="
    )
    assert "scale_context" not in inspect.signature(
        RIGNO._call_with_processed_rnodes
    ).parameters
    assert "scale_context" not in inspect.signature(
        RIGNO._apply_global_film
    ).parameters

    runtime: dict[str, Any] = {}
    if not args.skip_runtime:
        runtime["V38"] = run_smoke(
            BASELINE, batch_size=2, grid=(3, 3, 3)
        )
        assert runtime["V38"]["finite_forward_backward"]
        for name, path in CONFIGS.items():
            runtime[name] = run_smoke(path, batch_size=2, grid=(3, 3, 3))
            assert runtime[name]["finite_forward_backward"]
            assert runtime[name]["all_parameters_trainable"]
            assert all(
                count > 0
                for count in runtime[name][
                    "parameter_count_by_trainable_group"
                ].values()
            )
            assert all(
                value >= 0.0
                for value in runtime[name]["gradient_norms"].values()
            )
        assert runtime["V42"]["parameter_count"] == runtime["V38"]["parameter_count"]
        assert runtime["V43"]["parameter_count"] - runtime["V38"]["parameter_count"] == 640
        assert runtime["V44"]["parameter_count"] - runtime["V43"]["parameter_count"] == 28896
        assert runtime["V44"]["deepsets_output_init_max_abs"] == 0.0
        assert (
            runtime["V44"]["initial_prediction_checksums"]
            == runtime["V43"]["initial_prediction_checksums"]
        )

    rows = list(csv.DictReader(REGISTRY.open(encoding="utf-8", newline="")))
    assert [row["config_id"] for row in rows] == [
        configs[name]["config_id"] for name in ("V42", "V43", "V44")
    ]
    for row in rows:
        assert row["launch_policy"] == "explicit_user_instruction_only"
        assert row["execution_status"] == "not_started"
        assert row["training_started"] == "false"
        assert row["forbidden_access_roles"].split("|") == sorted(FORBIDDEN)
        assert row["smoke_status"] == "passed_real_B28_update"
        assert (
            row["smoke_commit"]
            == "42d4abe8dddf31fb25ab4764ccf1b31901b1025b"
        )
        assert float(row["smoke_peak_GiB"]) > 0.0

    committed = json.loads(REPORT.read_text(encoding="utf-8"))
    assert committed["training_started"] is False
    assert committed["formal_e600_started"] is False
    assert committed["resolved_config_diffs"] == diffs
    assert committed["expected_parameter_increment"]["V43_vs_V38"] == 640
    assert committed["expected_parameter_increment"]["V44_vs_V43"] == 28896
    real_smoke = committed["real_train_update_smoke"]
    assert real_smoke["status"] == "passed"
    assert real_smoke["batch_size"] == 28
    assert real_smoke["node_count"] == 1024
    assert real_smoke["accessed_roles"] == ["train"]
    assert real_smoke["forbidden_roles_accessed"] == []
    assert real_smoke["random_initialization"] is True
    assert real_smoke["checkpoint_written"] is False
    assert real_smoke["formal_e600_started"] is False
    smoke_results = real_smoke["results"]
    assert set(smoke_results) == {"V42", "V43", "V44"}
    for result in smoke_results.values():
        assert result["finite_loss"]
        assert result["finite_gradients"]
        assert result["finite_updated_parameters"]
        assert result["optimizer_update_applied"]
        assert result["update_nonzero"]
        assert result["peak_bytes_in_use"] > 0
    assert (
        smoke_results["V43"]["parameter_count"]
        - smoke_results["V42"]["parameter_count"]
        == 640
    )
    assert (
        smoke_results["V44"]["parameter_count"]
        - smoke_results["V43"]["parameter_count"]
        == 28896
    )
    partition = smoke_results["V44"]["p2r_partition_of_unity"]
    assert partition["zero_degree_node_count"] == 0
    assert partition["maximum_partition_of_unity_error"] <= 1.0e-12
    assert partition["source_conserved"]
    assert partition["volume_conserved"]
    reference = committed["V42_train_only_objective_reference"]
    assert reference["fit_roles"] == ["train"]
    assert reference["fit_sample_count"] == 672
    assert reference["raw_train_target_energy_per_point_K2"] > 0.0
    assert set(reference["raw_scale_weight_quantiles"]) == {
        "p00", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "p100"
    }
    assert 0.0 <= reference["lower_clip_fraction"] <= 1.0
    assert 0.0 <= reference["upper_clip_fraction"] <= 1.0
    assert 0.0 < reference["effective_sample_size"] <= 672.0
    print(
        json.dumps(
            {
                "status": "passed",
                "resolved_config_diffs": diffs,
                "runtime": runtime,
                "training_started": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
