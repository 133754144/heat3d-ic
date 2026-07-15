#!/usr/bin/env python3
"""Gate 6G attention, gradient-route, registry and resolved-config checks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shlex
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.heat3d_v2_config import validate_v2_config  # noqa: E402
from rigno.heat3d_v2_runner_command import build_training_command  # noqa: E402
from rigno.heat3d_v5_global_context import GLOBAL_CONTEXT_FEATURES  # noqa: E402
from rigno.heat3d_v5_scale_pooling import QK_REGION_FEATURES  # noqa: E402
from rigno.models.operator import Inputs  # noqa: E402
from rigno.models.rigno import RIGNO  # noqa: E402
from scripts.check_heat3d_v4_registry import resolve_inherited_yaml  # noqa: E402


REGISTRY = ROOT / "configs/heat3d_v5/v5_gate6g_attention_registry.csv"
GATE6E_REGISTRY = ROOT / "configs/heat3d_v5/v5_gate6e_scratch_loss_registry.csv"
V13 = ROOT / "configs/heat3d_v5/generated/V4P5_13_gate6e_scratch_branch_rebalance.yaml"
V13_CLOSEOUT = ROOT / "configs/heat3d_v5/gate6g/v13_closeout.json"
E1_SUMMARY = ROOT / "configs/heat3d_v5/gate6g/e1_smoke_summary.json"
IDS = (
    "V4P5_22_gate6g_control_constlr",
    "V4P5_23_gate6g_stopgrad_constlr",
    "V4P5_24_gate6g_shape_attention_constlr",
    "V4P5_25_gate6g_scale_attention_constlr",
    "V4P5_26_gate6g_shape_attention_stopgrad_constlr",
    "V4P5_27_gate6g_deep_scale_head_constlr",
)
FORBIDDEN = "test_iid|hard_train_holdout|hard_challenge_valid|hard_challenge_test|sealed_iid"
EXPECTED_VARIANTS = {
    IDS[0]: {},
    IDS[1]: {"model.pooled_latent_stop_gradient": True},
    IDS[2]: {"model.shape_attention_mode": "physics_gate"},
    IDS[3]: {
        "model.pooled_latent_stop_gradient": True,
        "model.scale_attention_mode": "physics_gate",
    },
    IDS[4]: {
        "model.pooled_latent_stop_gradient": True,
        "model.shape_attention_mode": "physics_gate",
    },
    IDS[5]: {"model.scale_head_depth": 3},
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--skip-runtime", action="store_true")
    return parser.parse_args()


def _resolved(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    resolved = resolve_inherited_yaml(payload, path)
    validate_v2_config(resolved, config_path=path)
    return resolved


def _get(payload: dict[str, Any], dotted: str) -> Any:
    value: Any = payload
    for part in dotted.split("."):
        value = value[part]
    return value


def _flatten(payload: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(payload, dict):
        result: dict[str, Any] = {}
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(value, path))
        return result
    return {prefix: payload}


def _scientific_diff(control: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    left, right = _flatten(control), _flatten(candidate)
    identity_prefixes = ("metadata.",)
    identity_fields = {
        "config_id",
        "description",
        "export.output_dir",
        "export.run_name",
        "run.memory_audit_jsonl",
        "run.final_probe_output_dir",
        "run.post_training_diagnostics_output_dir",
    }
    return {
        key: right.get(key)
        for key in sorted(set(left) | set(right))
        if left.get(key) != right.get(key)
        and key not in identity_fields
        and not key.startswith(identity_prefixes)
    }


def _tree_map(params: Any) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for path, value in jax.tree_util.tree_flatten_with_path(params)[0]:
        name = "/".join(str(getattr(item, "key", getattr(item, "name", item))) for item in path)
        result[name] = np.asarray(value)
    return result


def _param_count(params: Any) -> int:
    return int(sum(np.asarray(value).size for value in jax.tree_util.tree_leaves(params)))


def _group_norm(params: Any, predicate) -> float:
    total = 0.0
    for path, value in _tree_map(params).items():
        if predicate(path):
            total += float(np.sum(np.square(value)))
    return float(np.sqrt(total))


def _fixture(batch_size: int = 2) -> dict[str, Any]:
    axis = np.linspace(0.0, 1.0, 3, dtype=np.float32)
    coords = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    builder = Heat3DGraphBuilder(rmesh_levels=1, subsample_factor=2)
    metadata = builder.build_metadata(coords, key=jax.random.PRNGKey(11))
    metadata = jax.tree_util.tree_map(
        lambda value: jnp.repeat(value, repeats=batch_size, axis=0), metadata
    )
    graphs = builder.build_graphs(metadata)
    normalized = jnp.asarray(2.0 * coords - 1.0)[None, None, :, :]
    normalized = jnp.repeat(normalized, repeats=batch_size, axis=0)
    node = jnp.linspace(0.0, 1.0, coords.shape[0])[None, None, :, None]
    condition = jnp.concatenate([node + 0.1 * index for index in range(8)], axis=-1)
    condition = jnp.repeat(condition, repeats=batch_size, axis=0)
    inputs = Inputs(
        u=jnp.zeros((batch_size, 1, coords.shape[0], 1), dtype=jnp.float32),
        c=condition,
        x_inp=normalized,
        x_out=normalized,
        t=None,
        tau=None,
    )
    volumes = jnp.ones((batch_size, coords.shape[0]), dtype=jnp.float32)
    bottom = (coords[:, 2] == coords[:, 2].min()).astype(np.float32)
    mask = jnp.repeat(jnp.asarray(bottom)[None, :], repeats=batch_size, axis=0)
    context = jnp.linspace(
        -0.2, 0.2, batch_size * len(GLOBAL_CONTEXT_FEATURES), dtype=jnp.float32
    ).reshape(batch_size, len(GLOBAL_CONTEXT_FEATURES))
    log_s_phys = jnp.log(jnp.linspace(1.0, 1.2, batch_size, dtype=jnp.float32))
    reference = jnp.full((batch_size, coords.shape[0]), 300.0, dtype=jnp.float32)
    regional_count = int(metadata.x_rnodes.shape[1] - 1)
    qk = jnp.linspace(
        -0.3,
        0.4,
        batch_size * regional_count * len(QK_REGION_FEATURES),
        dtype=jnp.float32,
    ).reshape(batch_size, regional_count, len(QK_REGION_FEATURES))
    return {
        "inputs": inputs,
        "graphs": graphs,
        "volumes": volumes,
        "mask": mask,
        "context": context,
        "log_s_phys": log_s_phys,
        "reference": reference,
        "qk": qk,
    }


def _model(**overrides: Any) -> RIGNO:
    options = dict(
        num_outputs=1,
        node_latent_size=8,
        edge_latent_size=8,
        processor_steps=1,
        mlp_hidden_layers=1,
        concatenate_t=False,
        concatenate_tau=False,
        conditioned_normalization=False,
        p_edge_masking=0.0,
        decoder_bypass_mode="post_decoder_residual",
        decoder_bypass_features="explicit_local_condition",
        decoder_bypass_feature_indices=tuple(range(8)),
        decoder_bypass_feature_names=tuple(f"local_{index}" for index in range(8)),
        decoder_bypass_local_feature_names=tuple(f"local_{index}" for index in range(8)),
        decoder_bypass_num_features=8,
        decoder_bypass_output_space="native_psi",
        decoder_bypass_hidden_size=8,
        decoder_bypass_layers=1,
        decoder_bypass_init="zero_residual",
        global_context_mode="film",
        global_context_feature_dim=len(GLOBAL_CONTEXT_FEATURES),
        global_context_feature_names=tuple(GLOBAL_CONTEXT_FEATURES),
        film_target="rnodes_processed",
        film_init="identity",
        native_output_mode="native_shape_scale",
        native_branch_mode="joint",
        scale_head_mode="physics_plus_pooled_latent",
        scale_pooling="mean",
        scale_head_hidden_size=8,
    )
    options.update(overrides)
    return RIGNO(**options)


def _init_apply(model: RIGNO, fixture: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    kwargs = dict(
        inputs=fixture["inputs"],
        graphs=fixture["graphs"],
        control_volumes=fixture["volumes"],
        log_s_phys=fixture["log_s_phys"],
        reference_temperature=fixture["reference"],
        dirichlet_mask=fixture["mask"],
        prescribed_temperature=fixture["reference"],
        global_context=fixture["context"],
        qk_region_features=fixture["qk"],
        method=model.predict_native_shape_scale,
    )
    params = model.init(jax.random.PRNGKey(17), **kwargs)["params"]
    prediction = model.apply({"params": params}, **kwargs)
    return params, prediction


def _max_output_diff(left: dict[str, Any], right: dict[str, Any]) -> float:
    keys = ("psi", "phi_hat", "s_hat", "log_s_hat", "deltaT_hat", "pooled_rnodes")
    return max(float(jnp.max(jnp.abs(left[key] - right[key]))) for key in keys)


def _runtime_checks() -> dict[str, Any]:
    fixture = _fixture()
    default_model = _model()
    explicit_model = _model(
        shape_attention_mode="none",
        scale_attention_mode="none",
        regional_attention_hidden_size=64,
        pooled_latent_stop_gradient=False,
        scale_head_depth=1,
    )
    default_params, default_prediction = _init_apply(default_model, fixture)
    explicit_params, explicit_prediction = _init_apply(explicit_model, fixture)
    default_map, explicit_map = _tree_map(default_params), _tree_map(explicit_params)
    assert default_map.keys() == explicit_map.keys()
    assert max(float(np.max(np.abs(default_map[key] - explicit_map[key]))) for key in default_map) == 0.0
    assert _max_output_diff(default_prediction, explicit_prediction) == 0.0

    reports: dict[str, Any] = {}
    for name, options in {
        "control": {},
        "shape_attention": {"shape_attention_mode": "physics_gate"},
        "scale_attention": {"scale_attention_mode": "physics_gate"},
        "shape_attention_stopgrad": {
            "shape_attention_mode": "physics_gate",
            "pooled_latent_stop_gradient": True,
        },
        "deep_scale_head": {"scale_head_depth": 3},
    }.items():
        model = _model(**options)
        params, prediction = _init_apply(model, fixture)
        shared = set(default_map) & set(_tree_map(params))
        shared_max = max(
            (float(np.max(np.abs(default_map[key] - _tree_map(params)[key]))) for key in shared),
            default=0.0,
        )
        reports[name] = {
            "parameter_count": _param_count(params),
            "added_parameters": _param_count(params) - _param_count(default_params),
            "shared_parameter_path_count": len(shared),
            "shared_parameter_max_abs_difference": shared_max,
            "initial_output_max_abs_difference": _max_output_diff(default_prediction, prediction),
        }
        assert shared_max == 0.0
        if name in {"shape_attention", "scale_attention", "shape_attention_stopgrad"}:
            assert reports[name]["initial_output_max_abs_difference"] == 0.0
    assert np.array_equal(
        np.asarray(default_prediction["s_hat"]),
        np.asarray(_init_apply(_model(shape_attention_mode="physics_gate"), fixture)[1]["s_hat"]),
    )

    stop_model = _model(pooled_latent_stop_gradient=True)
    stop_params, _ = _init_apply(stop_model, fixture)

    def scale_loss(params):
        kwargs = dict(
            inputs=fixture["inputs"], graphs=fixture["graphs"],
            control_volumes=fixture["volumes"], log_s_phys=fixture["log_s_phys"],
            reference_temperature=fixture["reference"], dirichlet_mask=fixture["mask"],
            prescribed_temperature=fixture["reference"], global_context=fixture["context"],
            qk_region_features=fixture["qk"], method=stop_model.predict_native_shape_scale,
        )
        return jnp.sum(stop_model.apply({"params": params}, **kwargs)["log_s_hat"])

    gradients = jax.grad(scale_loss)(stop_params)
    is_scale = lambda path: any(token in path for token in ("global_scale_", "scale_attention"))
    backbone_norm = _group_norm(gradients, lambda path: not is_scale(path))
    scale_norm = _group_norm(gradients, is_scale)
    assert backbone_norm == 0.0
    assert scale_norm > 0.0
    assert not any("target" in name.lower() or "temperature" in name.lower() for name in QK_REGION_FEATURES)
    return {
        "default_disabled_parameter_paths_equal": True,
        "default_disabled_output_max_abs_difference": 0.0,
        "attention_reports": reports,
        "shape_attention_scale_path_max_abs_difference": 0.0,
        "stop_gradient_scale_loss_backbone_gradient_norm": backbone_norm,
        "stop_gradient_scale_head_gradient_norm": scale_norm,
        "target_leakage": False,
    }


def main() -> int:
    args = _args()
    rows = list(csv.DictReader(REGISTRY.open(encoding="utf-8", newline="")))
    assert tuple(row["config_id"] for row in rows) == IDS
    v13 = _resolved(V13)
    # V13's tracked inherited YAML predates explicit lr_peak serialization;
    # the frozen 2e-4 value is read from its WSL2 run_config in the closeout.
    commands: dict[str, str] = {}
    resolved_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        path = ROOT / row["generated_yaml"]
        resolved = _resolved(path)
        resolved_by_id[row["config_id"]] = resolved
        assert resolved["dataset"] == v13["dataset"]
        assert resolved["graph"] == v13["graph"]
        assert resolved["loss"] == v13["loss"]
        assert resolved["run"]["epochs"] == 200
        assert resolved["run"]["batch_size"] == 28
        assert resolved["run"]["init_checkpoint"] is None
        assert resolved["optimizer"]["multi_seed"] == []
        assert resolved["optimizer"]["lr_schedule"] == "constant"
        assert float(resolved["optimizer"]["lr"]) == 1.0e-4
        assert float(row["constant_lr"]) == 0.5 * float(row["v13_lr_peak"])
        assert resolved["export"]["prediction_split"] == "valid_iid"
        assert resolved["export"]["selection_metric"] == "valid_rel_rmse_v4_pct"
        assert resolved["export"]["save_point_global_best_checkpoint"] is True
        assert resolved["export"]["save_base_mse_best_checkpoint"] is True
        assert resolved["export"]["save_sample_first_best_checkpoint"] is True
        assert row["fit_roles"] == row["normalization_fit_roles"] == "train"
        assert row["selection_roles"] == "valid_iid"
        assert row["forbidden_access_roles"] == FORBIDDEN
        assert row["long_training_started"] == "false"
        for field, expected in EXPECTED_VARIANTS[row["config_id"]].items():
            assert _get(resolved, field) == expected
        command = build_training_command(resolved, python_executable="python")
        text = shlex.join(command)
        assert "--epochs 200" in text and "--batch-size 28" in text
        assert "--lr 0.0001" in text and "--lr-schedule constant" in text
        assert "--prediction-split valid_iid" in text
        assert "--init-checkpoint" not in command
        commands[row["config_id"]] = text

    control = resolved_by_id[IDS[0]]
    for config_id in IDS[1:]:
        diff = _scientific_diff(control, resolved_by_id[config_id])
        assert diff == EXPECTED_VARIANTS[config_id], (config_id, diff)

    with GATE6E_REGISTRY.open(encoding="utf-8", newline="") as handle:
        gate6e_rows = list(csv.DictReader(handle))
    assert len(gate6e_rows) == 1
    gate6e = gate6e_rows[0]
    assert gate6e["plan_status"] == "completed"
    assert gate6e["execution_status"] == "completed_e600"
    assert gate6e["evaluation_status"] == "completed_valid_only_closeout"
    assert gate6e["training_started"] == "true"
    closeout = json.loads(V13_CLOSEOUT.read_text(encoding="utf-8"))
    assert closeout["status"] == "completed"
    assert closeout["split"]["roles_read_for_closeout"] == ["valid_iid"]
    assert closeout["split"]["forbidden_roles_read"] == []
    assert closeout["train_only_standardizer"]["fit_population"] == "train_only"
    assert closeout["train_only_standardizer"]["fit_sample_count"] == 672
    assert closeout["selection_records"]["base_mse_best"]["checkpoint_saved"] is True
    for name in ("point_global_best", "sample_first_best"):
        assert closeout["selection_records"][name]["trajectory_only"] is True
        assert closeout["selection_records"][name]["checkpoint_saved"] is False
    assert closeout["model_inference_run"] is False
    assert closeout["run_training_completed"] is True
    assert closeout["closeout_training_started"] is False
    assert closeout["large_artifacts_tracked"] is False
    assert all(len(row["sha256"]) == 64 for row in closeout["artifacts"].values())

    e1 = None
    if E1_SUMMARY.is_file():
        e1 = json.loads(E1_SUMMARY.read_text(encoding="utf-8"))
        assert e1["status"] == "completed"
        assert [row["config_id"] for row in e1["results"]] == list(IDS)
        assert e1["roles_accessed"] == ["train", "valid_iid"]
        assert e1["forbidden_roles_accessed"] == []
        assert e1["sealed_iid_accessed"] is False
        assert e1["long_training_started"] is False
        for row in e1["results"]:
            assert row["status"] == "passed"
            assert row["nodes_per_sample"] == 1024 and row["batch_size"] == 28
            assert row["checkpoint_reload_passed"] is True
            assert row["checkpoint_reload_entry_count"] >= 5
            assert row["grad_finite"] is True
            assert row["parameter_count"] > 0 and row["peak_rss_mb"] > 0.0

    runtime = None if args.skip_runtime else _runtime_checks()
    payload = {
        "status": "passed",
        "config_count": len(rows),
        "v13_lr_peak": 2.0e-4,
        "constant_lr": 1.0e-4,
        "constant_lr_formula": "0.5 * V13 run_config lr_peak",
        "roles_accessed": [],
        "forbidden_roles_accessed": [],
        "sealed_iid_accessed": False,
        "long_training_started": False,
        "runtime": runtime,
        "v13_closeout_checked": True,
        "e1_smoke_checked": e1 is not None,
        "commands": commands,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
