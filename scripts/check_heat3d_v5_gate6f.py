#!/usr/bin/env python3
"""Static Gate 6F registry, runner and low-memory control checks."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import shlex
import sys
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rigno.heat3d_v2_config import validate_v2_config  # noqa: E402
from rigno.heat3d_v2_runner_command import build_training_command  # noqa: E402
from rigno.heat3d_v5_scale_pooling import (  # noqa: E402
    QK_REGION_FEATURES,
    SCALE_POOLING_MODES,
    qk_region_features_from_raw,
)
from rigno.models.rigno import RIGNO  # noqa: E402
from scripts.check_heat3d_v4_registry import resolve_inherited_yaml  # noqa: E402


REGISTRY = ROOT / "configs/heat3d_v5/v5_gate6f_scale_probe_registry.csv"
N3 = ROOT / "configs/heat3d_v5/generated/V4P5_07_native_pooled_latent_global_film.yaml"
AMPLITUDE_REPORT = ROOT / "configs/heat3d_v5/gate6f/amplitude_valid_only.json"
FROZEN_CACHE_MANIFEST = ROOT / "configs/heat3d_v5/gate6f/frozen_feature_cache_manifest.json"
FROZEN_PROBE_REPORT = ROOT / "configs/heat3d_v5/gate6f/frozen_probe_screen_lowmem.json"
E1_SMOKE_SUMMARY = ROOT / "configs/heat3d_v5/gate6f/e1_smoke_summary.json"
GATE6F_CLOSEOUT = ROOT / "configs/heat3d_v5/gate6f/gate6f_closeout.json"
EXPECTED_IDS = tuple(f"V4P5_{index:02d}_gate6f_{suffix}_smoke" for index, suffix in (
    (14, "mean_pool"),
    (15, "mean_std"),
    (16, "mean_max"),
    (17, "pre_film_mean_std"),
    (18, "deep_scale_head"),
    (19, "latent_attention"),
    (20, "qk_gated"),
    (21, "mean_decoupled"),
))
FORBIDDEN = "test_iid|hard_train_holdout|hard_challenge_valid|hard_challenge_test"
EXPECTED_FROZEN_RANKS = {
    "V4P5_14_gate6f_mean_pool_smoke": "1",
    "V4P5_15_gate6f_mean_std_smoke": "4",
    "V4P5_16_gate6f_mean_max_smoke": "3",
    "V4P5_17_gate6f_pre_film_mean_std_smoke": "7",
    "V4P5_18_gate6f_deep_scale_head_smoke": "2",
    "V4P5_19_gate6f_latent_attention_smoke": "6",
    "V4P5_20_gate6f_qk_gated_smoke": "5",
    "V4P5_21_gate6f_mean_decoupled_smoke": "",
}


def _resolved(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    resolved = resolve_inherited_yaml(payload, path)
    validate_v2_config(resolved, config_path=path)
    return resolved


def _assert_qk_input_only() -> None:
    names = (
        "k_z", "q", "is_top", "is_bottom", "is_side", "is_interior",
        "top_h", "bottom_T_fixed_minus_T_ref",
    )
    coords = np.asarray([[0.0, 0.0, 0.0], [0.2, 0.0, 0.5], [0.8, 0.0, 0.7], [1.0, 0.0, 1.0]])
    raw = np.asarray([
        [2.0, 0.0, 1, 0, 1, 0, 1000.0, 0.0],
        [1.0, 3.0, 0, 0, 0, 1, 1000.0, 0.0],
        [0.5, 5.0, 0, 0, 0, 1, 1000.0, 0.0],
        [1.5, 0.0, 0, 1, 1, 0, 1000.0, 5.0],
    ])
    region = qk_region_features_from_raw(
        coords=coords,
        raw_condition=raw,
        condition_feature_names=names,
        p2r_edge_indices=np.asarray([[0, 0], [1, 0], [2, 1], [3, 1], [4, 2]]),
        rnode_count=2,
    )
    assert region.shape == (2, len(QK_REGION_FEATURES))
    assert np.all(np.isfinite(region))


def _assert_frozen_result_contracts() -> None:
    amplitude = json.loads(AMPLITUDE_REPORT.read_text(encoding="utf-8"))
    assert amplitude["roles_accessed"] == ["valid_iid"]
    assert amplitude["forbidden_roles_accessed"] == []
    assert amplitude["sealed_iid_accessed"] is False
    assert amplitude["training_started"] is False
    assert amplitude["checkpoint_inference_run"] is False
    assert amplitude["models"]["n3"]["checkpoint_epoch"] == 402
    assert amplitude["models"]["l2"]["checkpoint_epoch"] == 353
    assert set(amplitude["bins"]) == {
        "true_cv_rms_deltaT_K",
        "q_low_k_overlap_fraction",
        "q_weighted_inverse_conductivity_mK_W",
        "total_power_W",
        "source_concentration",
    }

    cache = json.loads(FROZEN_CACHE_MANIFEST.read_text(encoding="utf-8"))
    assert cache["config_id"] == "V4P5_07_native_pooled_latent_global_film"
    assert cache["checkpoint_kind"] == "best" and cache["checkpoint_epoch"] == 402
    assert cache["roles_materialized"] == ["train", "valid_iid"]
    assert cache["forbidden_roles_materialized"] == []
    assert cache["sealed_iid_accessed"] is False
    assert cache["training_started"] is False and cache["gnn_backward"] is False
    assert cache["global_context_standardizer"]["fit_population"] == "train_only"
    assert cache["default_disabled_control_replay"]["passed"] is True
    assert cache["default_disabled_control_replay"]["parameter_max_abs_difference"] == 0.0
    assert max(
        cache["default_disabled_control_replay"]["output_max_abs_difference"].values()
    ) == 0.0
    required_arrays = {
        "global_context", "global_context_raw", "rnodes_processed_pre_film",
        "rnodes_processed", "qk_region_features", "phi_hat", "s_phys", "s_true",
    }
    assert cache["splits"]["train"]["sample_count"] == 672
    assert cache["splits"]["valid_iid"]["sample_count"] == 128
    for split in ("train", "valid_iid"):
        record = cache["splits"][split]
        assert required_arrays <= set(record["array_shapes"])
        assert len(record["artifact_sha256"]) == 64
        assert len(record["sample_ids_sha256"]) == 64

    probes = json.loads(FROZEN_PROBE_REPORT.read_text(encoding="utf-8"))
    assert probes["roles_accessed"] == ["train", "valid_iid"]
    assert probes["forbidden_roles_accessed"] == []
    assert probes["sealed_iid_accessed"] is False
    assert probes["gnn_backward"] is False
    assert probes["frozen_probe_short_training"] is True
    assert probes["xla_python_client_preallocate"] == "false"
    assert probes["ranking"] == [
        "mean", "deep_scale_head", "mean_plus_max", "mean_plus_std",
        "qk_gated_pooling", "latent_attention_pooling", "pre_film_mean_plus_std",
    ]
    assert len(probes["results"]) == 7
    for result in probes["results"]:
        assert result["finite"] is True and result["gnn_backward"] is False
        assert int(result["parameter_count"]) > 0
        assert np.isfinite(float(result["valid"]["scale_log_rmse"]))
        assert np.isfinite(
            float(result["valid"]["fixed_shape_joint_point_global_relative_rmse_pct"])
        )


def _assert_e1_closeout(rows: list[dict[str, str]]) -> None:
    summary = json.loads(E1_SMOKE_SUMMARY.read_text(encoding="utf-8"))
    assert summary["config_count"] == 8
    assert summary["roles_accessed"] == ["train", "valid_iid"]
    assert summary["forbidden_roles_accessed"] == []
    assert summary["sealed_iid_accessed"] is False
    assert summary["long_training_started"] is False
    results = {result["config_id"]: result for result in summary["results"]}
    assert tuple(results) == EXPECTED_IDS
    for row in rows:
        result = results[row["config_id"]]
        assert result["status"] == row["e1_status"]
        assert result["status_ok"] is True
        assert result["grad_finite"] is True
        assert result["checkpoint_reload_passed"] is True
        assert result["native_runtime_passed"] is True
        assert result["roles_accessed"] == ["train", "valid_iid"]
        assert result["forbidden_roles_accessed"] == []
        assert result["sealed_iid_accessed"] is False
        assert np.isfinite(float(result["valid_base_mse"]))
        assert np.isfinite(float(result["valid_point_global_relative_rmse_pct"]))
        assert float(result["peak_rss_mb"]) == float(row["e1_peak_rss_mb"])
        assert float(result["peak_device_memory_mb"]) == float(
            row["e1_peak_device_memory_mb"]
        )
    assert results["V4P5_16_gate6f_mean_max_smoke"]["training_restarted"] is False

    closeout = json.loads(GATE6F_CLOSEOUT.read_text(encoding="utf-8"))
    assert closeout["status"] == "completed_low_memory_pre_research"
    assert closeout["execution_host"] == "devbox"
    assert closeout["wsl2_connected"] is False
    assert closeout["v13_modified_or_interfered"] is False
    assert closeout["roles_materialized"] == ["train", "valid_iid"]
    assert closeout["forbidden_roles_materialized"] == []
    assert closeout["sealed_iid_accessed"] is False
    assert closeout["long_training_started"] is False
    assert closeout["e600_started"] is False
    assert closeout["multi_seed_started"] is False
    assert closeout["e1_smoke"]["passed_count"] == 8
    assert closeout["e1_smoke"]["v16_training_restarted"] is False


def main() -> int:
    rows = list(csv.DictReader(REGISTRY.open(encoding="utf-8", newline="")))
    assert tuple(row["config_id"] for row in rows) == EXPECTED_IDS
    assert len({row["output_dir"] for row in rows}) == len(rows)
    assert len({row["memory_audit_jsonl"] for row in rows}) == len(rows)
    n3 = _resolved(N3)
    commands: dict[str, str] = {}
    for row in rows:
        path = ROOT / row["generated_yaml"]
        resolved = _resolved(path)
        assert row["baseline_config_id"] == "V4P5_07_native_pooled_latent_global_film"
        assert resolved["dataset"] == n3["dataset"]
        assert resolved["graph"] == n3["graph"]
        assert resolved["run"]["epochs"] == int(row["epochs"]) == 1
        assert resolved["run"]["batch_size"] == 28
        assert resolved["run"]["validation_batch_size"] == 128
        assert resolved["run"]["init_checkpoint"] is None
        assert resolved["optimizer"]["multi_seed"] == []
        assert resolved["export"]["prediction_split"] == "valid_iid"
        assert resolved["export"]["selection_metric"] == "valid_base_mse"
        assert row["fit_roles"] == row["normalization_fit_roles"] == "train"
        assert row["selection_roles"] == "valid_iid"
        assert row["forbidden_access_roles"] == FORBIDDEN
        assert row["launch_policy"] == "explicit_user_instruction_only"
        assert row["long_training_started"] == "false"
        assert row["frozen_probe_rank"] == EXPECTED_FROZEN_RANKS[row["config_id"]]
        assert row["plan_status"] == "frozen_prepared"
        assert row["execution_status"] in {"not_started", "completed_e1_smoke"}
        assert row["evaluation_status"] in {"not_evaluated", "completed_smoke"}
        if row["execution_status"] == "not_started":
            assert row["evaluation_status"] == "not_evaluated"
            assert row["e1_status"] == ""
            assert row["e1_peak_rss_mb"] == ""
            assert row["e1_peak_device_memory_mb"] == ""
        else:
            assert row["evaluation_status"] == "completed_smoke"
            assert row["e1_status"] in {"passed", "passed_recovered_post_interrupt"}
            if row["e1_status"] == "passed_recovered_post_interrupt":
                assert row["config_id"] == "V4P5_16_gate6f_mean_max_smoke"
            assert float(row["e1_peak_rss_mb"]) > 0.0
            # JAX exposes no allocator stats on some GPU builds.  Preserve that
            # explicit unavailability instead of treating it as a failed smoke.
            assert row["e1_peak_device_memory_mb"] in {"", "unavailable"} or float(
                row["e1_peak_device_memory_mb"]
            ) >= 0.0
        assert bool(resolved["metadata"]["long_training_started"]) is False
        assert bool(resolved["metadata"]["e600_started"]) is False
        assert bool(resolved["metadata"]["e600_completed"]) is False
        assert resolved["metadata"]["registry_config_id"] == row["config_id"]
        assert str(resolved["export"]["output_dir"]) == row["output_dir"]
        assert str(resolved["run"]["memory_audit_jsonl"]) == row["memory_audit_jsonl"]
        assert not (ROOT / row["output_dir"]).exists()
        assert not (ROOT / row["memory_audit_jsonl"]).exists()
        assert resolved["model"]["scale_pooling"] == row["scale_pooling"]
        assert int(resolved["model"]["scale_head_depth"]) == int(row["scale_head_depth"])
        assert bool(resolved["model"]["pooled_latent_stop_gradient"]) == (
            row["pooled_latent_stop_gradient"] == "true"
        )
        assert float(resolved["optimizer"]["scale_head_lr_multiplier"]) == float(
            row["scale_head_lr_multiplier"]
        )
        if row["scale_pooling"] == "qk_gated":
            assert row["qk_input_provenance"] == "raw_coords_k_q_bc_only"
            assert resolved["metadata"]["qk_region_feature_schema"] == "heat3d_v5_qk_region_features_v1"
        command = build_training_command(resolved, python_executable="python")
        text = shlex.join(command)
        commands[row["config_id"]] = text
        assert "--epochs 1" in text and "--batch-size 28" in text
        assert "--prediction-split valid_iid" in text
        assert "--init-checkpoint" not in command
        assert "--scale-pooling " + row["scale_pooling"] in text
        assert "--scale-head-depth " + row["scale_head_depth"] in text
        if row["pooled_latent_stop_gradient"] == "true":
            assert "--pooled-latent-stop-gradient" in command
        else:
            assert "--no-pooled-latent-stop-gradient" in command
        assert "--scale-head-lr-multiplier " + row["scale_head_lr_multiplier"] in text
    assert set(SCALE_POOLING_MODES) == {
        "mean", "mean_std", "mean_max", "pre_film_mean_std", "latent_attention", "qk_gated"
    }
    assert RIGNO.scale_head_depth == 1
    assert RIGNO.pooled_latent_stop_gradient is False
    # The N3 YAML leaves all Gate 6F controls absent, therefore model defaults
    # retain the old parameter graph and output route.  The cache exporter
    # performs the corresponding exact parameter/output replay check on devbox.
    assert "scale_head_depth" not in n3["model"]
    assert "pooled_latent_stop_gradient" not in n3["model"]
    assert "scale_head_lr_multiplier" not in n3["optimizer"]
    _assert_qk_input_only()
    _assert_frozen_result_contracts()
    _assert_e1_closeout(rows)
    v4_registry = (ROOT / "configs/heat3d_v4/run_registry.csv").read_text(encoding="utf-8")
    assert not any(config_id in v4_registry for config_id in EXPECTED_IDS)
    print(json.dumps({
        "status": "passed",
        "registry": str(REGISTRY),
        "config_count": len(rows),
        "pooling_modes": list(SCALE_POOLING_MODES),
        "qk_region_feature_count": len(QK_REGION_FEATURES),
        "e1_training_completed": True,
        "long_training_started": False,
        "commands": commands,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
